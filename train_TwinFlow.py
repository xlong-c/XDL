"""
TwinFlow 训练器 - 用于Diffusers模型的单步/少步生成加速
基于 xdl 训练框架，支持 TwinFlow 论文中的自对抗流训练

参考:
- TwinFlow论文: https://arxiv.org/abs/2512.05150
- 项目页: https://zhenglin-cheng.com/twinflow
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List, Optional, Dict, Any
import numpy as np

# 导入 xdl 训练框架
from xdl.trainer.coreModel import CoreModel
from xdl.trainer.trainer import Trainer
from xdl.callbacks.logging_callback import LoggingCallback
from xdl.callbacks.sampling_animation_callback import SamplingAnimationCallback

# 导入 TwinFlow 训练模块
from xdl.model.generate.twinflow import TwinFlow as TwinFlowTrainer


class DiffusionUNet(nn.Module):
    """
    简化的扩散模型 UNet，支持条件生成
    兼容 MNIST 数据集 (28x28)
    """
    
    def __init__(
        self,
        data_dim: int = 784,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        num_classes: int = 10,
        label_embed_dim: int = 32,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.side_len = int(np.sqrt(data_dim))  # MNIST: 28
        
        # 时间嵌入 (t 和 tt 分别嵌入)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        self.target_time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 标签嵌入 (条件)
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
        
        # 条件投影
        total_cond_dim = time_embed_dim * 2 + label_embed_dim
        self.cond_projection = nn.Linear(total_cond_dim, hidden_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, data_dim),
        )
        
    def forward(self, x, t, tt=None, c=None):
        """
        Args:
            x: 输入数据 [B, data_dim]
            t: 当前时间 [B] 或 [B, 1]
            tt: 目标时间 [B] 或 [B, 1] (可选)
            c: 条件标签列表 [B]
        Returns:
            output: 预测的速度场或噪声 [B, data_dim]
        """
        batch_size = x.size(0)
        
        # 处理时间输入
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [B, 1]
        t_emb = self.time_embedding(t)  # [B, time_embed_dim]
        
        # 处理目标时间
        if tt is not None:
            if tt.dim() == 1:
                tt = tt.unsqueeze(1)
            tt_emb = self.target_time_embedding(tt)
        else:
            tt_emb = torch.zeros_like(t_emb)
        
        # 处理条件
        if c is not None and len(c) > 0:
            labels = c[0]  # [B]
            label_emb = self.label_embedding(labels)  # [B, label_embed_dim]
        else:
            label_emb = torch.zeros(batch_size, self.label_embedding.embedding_dim).to(x.device)
        
        # 合并条件
        cond = torch.cat([t_emb, tt_emb, label_emb], dim=1)  # [B, total_cond_dim]
        cond_proj = self.cond_projection(cond)  # [B, hidden_dim]
        
        # 将条件与输入拼接
        x_cond = torch.cat([x, cond_proj], dim=1)  # [B, data_dim + hidden_dim]
        
        # Encoder
        h = self.encoder(x_cond)
        
        # Decoder
        output = self.decoder(h)
        
        return output


class TwinFlowCoreModel(CoreModel):
    """
    TwinFlow 核心模型类，继承自 CoreModel
    整合扩散模型和 TwinFlow 训练逻辑
    """
    
    def __init__(
        self,
        data_dim: int = 784,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        num_classes: int = 10,
        label_embed_dim: int = 32,
        # TwinFlow 参数
        ema_decay_rate: float = 0.99,
        estimate_order: int = 2,
        enhanced_ratio: float = 0.5,
        using_twinflow: bool = True,
        learning_rate: float = 2e-4,
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.learning_rate = learning_rate
        
        # 创建扩散模型
        self.model = DiffusionUNet(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_classes=num_classes,
            label_embed_dim=label_embed_dim,
        )
        
        # 创建 TwinFlow 训练器
        self.twinflow_trainer = TwinFlowTrainer(
            ema_decay_rate=ema_decay_rate,
            estimate_order=estimate_order,
            enhanced_ratio=enhanced_ratio,
            using_twinflow=using_twinflow,
        )
        
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        data, labels = batch
        device = next(self.parameters()).device
        
        # 准备数据
        data = data.to(device)
        if data.dim() > 2:
            # 将图像展平 [B, C, H, W] -> [B, C*H*W]
            data = data.view(data.size(0), -1)
        labels = labels.to(device)
        
        # 准备条件
        c = [labels]
        # 随机打乱标签作为无条件/负样本
        e = [labels[torch.randperm(labels.size(0))]]
        
        # 使用 TwinFlow 训练器计算损失
        loss = self.twinflow_trainer.training_step(
            model=self.model,
            x=data,
            c=c,
            e=e,
        )
        
        # 记录指标
        self.log('loss', loss.item())
        self.log('lr', self.get_lr())
        
        # 手动反向传播
        self.manual_backward(loss)
        
        # 梯度裁剪和优化器步进
        self.clip_gradients(
            model=self.model,
            gradient_clip_val=1.0,
        )
        
        for optimizer in self.optimizers:
            optimizer.step()
        
        for optimizer in self.optimizers:
            optimizer.zero_grad()
            
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        data, labels = batch
        device = next(self.parameters()).device
        
        data = data.to(device)
        if data.dim() > 2:
            data = data.view(data.size(0), -1)
        labels = labels.to(device)
        
        # 使用 TwinFlow 的采样方法生成样本
        with torch.no_grad():
            # 从噪声采样
            z = torch.randn_like(data)
            
            # 使用采样循环
            samples = self.twinflow_trainer.sampling_loop(
                inital_noise_z=z[:16],  # 只采样16个用于验证
                sampling_model=self.model,
                sampling_steps=1,  # 单步生成
                c=[labels[:16]],
            )
            
            # 计算重建损失
            generated = samples[-1]
            recon_loss = F.mse_loss(generated, data[:16])
            
        self.log('val_recon_loss', recon_loss.item())
        
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
        )
        return optimizer
        
    def generate_samples(self, n_samples: int = 16, labels: Optional[torch.Tensor] = None):
        """
        从潜在空间生成样本
        
        Args:
            n_samples: 采样数量
            labels: 条件标签，如果为None则随机生成
            
        Returns:
            samples: 生成的样本 [n_samples, data_dim]
        """
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # 从标准正态分布采样潜在变量
            z = torch.randn(n_samples, self.data_dim).to(device)
            
            # 准备条件
            if labels is None:
                labels = torch.randint(0, self.num_classes, (n_samples,)).to(device)
            else:
                labels = labels.to(device)
            
            # 使用采样循环生成
            samples = self.twinflow_trainer.sampling_loop(
                inital_noise_z=z,
                sampling_model=self.model,
                sampling_steps=1,  # 单步生成
                c=[labels],
            )
            
            return samples[-1]


def get_mnist_dataloaders(
    batch_size: int = 128,
    data_root: str = './data',
    num_workers: int = 2,
):
    """获取 MNIST 数据加载器"""
    
    # 数据预处理 - 归一化到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    # 训练集
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True,
    )
    
    # 验证集
    val_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True,
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def main():
    """主训练函数"""
    
    print("=" * 60)
    print("TwinFlow Training on MNIST")
    print("基于 xdl 训练框架")
    print("=" * 60)
    
    # ==================== 配置参数 ====================
    # 数据参数
    batch_size = 128
    data_root = './data'
    
    # 模型参数
    data_dim = 784  # 28x28
    hidden_dim = 128
    time_embed_dim = 64
    num_classes = 10
    label_embed_dim = 32
    
    # TwinFlow 训练参数
    ema_decay_rate = 0.99
    estimate_order = 2
    enhanced_ratio = 0.5
    using_twinflow = True
    learning_rate = 2e-4
    
    # 训练参数
    max_epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 日志参数
    log_dir = './others/logs/twinflow'
    checkpoint_dir = './others/checkpoints/twinflow'
    
    # ==================== 创建目录 ====================
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ==================== 数据准备 ====================
    print("\n[1/5] 加载 MNIST 数据集...")
    train_loader, val_loader = get_mnist_dataloaders(
        batch_size=batch_size,
        data_root=data_root,
    )
    print(f"      训练集: {len(train_loader.dataset)} 样本")
    print(f"      验证集: {len(val_loader.dataset)} 样本")
    
    # ==================== 模型创建 ====================
    print("\n[2/5] 创建 TwinFlow 模型...")
    model = TwinFlowCoreModel(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        num_classes=num_classes,
        label_embed_dim=label_embed_dim,
        ema_decay_rate=ema_decay_rate,
        estimate_order=estimate_order,
        enhanced_ratio=enhanced_ratio,
        using_twinflow=using_twinflow,
        learning_rate=learning_rate,
    )
    print(f"      模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== 回调函数 ====================
    print("\n[3/5] 配置回调函数...")
    
    # 日志回调
    logging_cb = LoggingCallback(
        log_frequency=50,
        log_dir=log_dir,
        log_filename='twinflow_training.log',
        rotation="10 MB",
        enable_console=True,
    )
    
    # 采样动画回调
    animation_cb = SamplingAnimationCallback(
        n_samples=16,
        save_dir='./others/animations/twinflow',
        animation_filename='twinflow_training.gif',
        animation_format='gif',
        fps=2,
        sample_method='generate_samples',
        save_intermediate_images=True,
        figsize=(10, 10),
        show_epoch_label=True,
    )
    
    callbacks = [logging_cb, animation_cb]
    print(f"      已配置 {len(callbacks)} 个回调函数")
    
    # ==================== 训练器 ====================
    print("\n[4/5] 初始化训练器...")
    trainer = Trainer(
        max_epochs=max_epochs,
        device=device,
        precision='bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '32',
        callbacks=callbacks,
    )
    
    # 配置日志和检查点
    trainer.setup_logger(
        experiment_name='twinflow_mnist',
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        monitor='val_recon_loss',
        mode='min',
        save_top_k=3,
        log_every_n_steps=50,
        enable_tqdm=True,
        enable_console=True,
        enable_tensorboard=True,
        enable_total_progress=True,
        tqdm_metric_keys=['loss', 'lr', 'val_recon_loss'],
    )
    
    print(f"      设备: {device}")
    print(f"      Epochs: {max_epochs}")
    print(f"      批次大小: {batch_size}")
    
    # ==================== 开始训练 ====================
    print("\n[5/5] 开始训练...")
    print("=" * 60)
    
    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        val_check_interval=500,  # 每500步验证一次
    )
    
    # ==================== 保存最终模型 ====================
    print("\n" + "=" * 60)
    print("训练完成！保存最终模型...")
    
    final_model_path = model.save_checkpoint(
        base_dir=checkpoint_dir,
        naming_keys=['final', 'epoch', 'step'],
        custom_values={'loss': trainer.callback_metrics.get('loss', 0)},
        save_optimizer=True,
    )
    
    print(f"模型已保存到: {final_model_path}")
    
    # ==================== 生成最终可视化 ====================
    print("\n生成最终样本...")
    model.eval()
    
    with torch.no_grad():
        # 生成每个数字的样本
        n_samples_per_class = 8
        all_samples = []
        
        for class_idx in range(10):
            labels = torch.full((n_samples_per_class,), class_idx, dtype=torch.long)
            samples = model.generate_samples(
                n_samples=n_samples_per_class,
                labels=labels,
            )
            all_samples.append(samples)
        
        # 合并所有样本
        all_samples = torch.cat(all_samples, dim=0)  # [80, 784]
        
        # 重塑为图像格式
        all_samples = all_samples.view(-1, 1, 28, 28)
        
        # 归一化到 [0, 1]
        all_samples = (all_samples + 1) / 2.0
        all_samples = all_samples.clamp(0, 1)
        
        # 保存图像网格
        from torchvision.utils import save_image
        save_path = os.path.join(checkpoint_dir, 'final_samples.png')
        save_image(all_samples, save_path, nrow=n_samples_per_class, padding=2)
        
        print(f"样本已保存到: {save_path}")
    
    print("\n" + "=" * 60)
    print("TwinFlow 训练完成！")
    print(f"日志目录: {log_dir}")
    print(f"检查点目录: {checkpoint_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
