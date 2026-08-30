#!/usr/bin/env python3
"""
GAN MNIST训练脚本
使用项目的trainer框架进行GAN模型的训练
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from xdl.callbacks.logging_callback import LoggingCallback
from xdl.callbacks.sampling_animation_callback import SamplingAnimationCallback

# 导入项目的trainer框架
from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Generator(nn.Module):
    """
    生成器网络 (Generator)
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers: list[nn.Module] = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    """
    判别器网络 (Discriminator)
    """
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GANModel(CoreModel):
    """
    GAN模型类, 继承自CoreModel
    """
    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
        
        self.loss_fn = nn.BCELoss()

    def forward(self, z):
        """前向传播 (仅用于生成)"""
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        """
        GAN的训练步骤: 交替更新生成器和判别器
        """
        imgs, _ = batch
        device = imgs.device
        
        # 获取优化器 (在configure_optimizers中返回了两个)
        optimizer_g, optimizer_d = self.optimizers
        
        # 地面真值标签 (真图为1, 假图为0)
        valid = torch.ones(imgs.size(0), 1, device=device)
        fake = torch.zeros(imgs.size(0), 1, device=device)

        # -----------------
        #  训练生成器 (Generator)
        # -----------------
        # 目标: 让判别器认为生成的图像是真的
        
        # 采样噪声
        z = torch.randn(imgs.size(0), self.latent_dim, device=device)
        
        # 生成图像
        gen_imgs = self.generator(z)
        
        # 判别器对生成图像的打分
        # 我们希望 discriminator(gen_imgs) 接近 valid (1)
        g_loss = self.loss_fn(self.discriminator(gen_imgs), valid)
        
        # 手动反向传播并更新生成器
        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()

        # ---------------------
        #  训练判别器 (Discriminator)
        # ---------------------
        # 目标: 准确区分真图和假图
        
        # 计算真实图像的损失 (真实图像标记为1)
        real_loss = self.loss_fn(self.discriminator(imgs), valid)
        # 计算伪造图像的损失 (生成图像标记为0)
        # 注意: 使用 gen_imgs.detach() 避免在训练判别器时梯度流向生成器
        fake_loss = self.loss_fn(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        # 手动反向传播并更新判别器
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()

        # 记录损失到日志系统
        self.log('g_loss', g_loss.item())
        self.log('d_loss', d_loss.item())
        self.log('d_real_loss', real_loss.item())
        self.log('d_fake_loss', fake_loss.item())

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        imgs, _ = batch
        device = imgs.device
        
        valid = torch.ones(imgs.size(0), 1, device=device)
        fake = torch.zeros(imgs.size(0), 1, device=device)
        
        z = torch.randn(imgs.size(0), self.latent_dim, device=device)
        gen_imgs = self.generator(z)
        
        g_loss = self.loss_fn(self.discriminator(gen_imgs), valid)
        real_loss = self.loss_fn(self.discriminator(imgs), valid)
        fake_loss = self.loss_fn(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        self.log('val_g_loss', g_loss.item())
        self.log('val_d_loss', d_loss.item())

    def configure_optimizers(self):
        """配置优化器: 返回生成器和判别器的优化器列表"""
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999
        
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        # 返回列表形式
        return [optimizer_g, optimizer_d]

    def generate_samples(self, n_samples: int = 16):
        """从噪声生成样本用于可视化"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            gen_imgs = self.generator(z)
            # MNIST数据归一化到了 [-1, 1], 需要转换回 [0, 1] 显示
            gen_imgs = (gen_imgs + 1) / 2
            return gen_imgs

def get_mnist_dataloaders(batch_size: int = 64):
    """获取MNIST数据加载器"""
    # GAN 常用 Tanh 激活, 所以数据归一化到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    os.makedirs('./others/data', exist_ok=True)
    dataset = datasets.MNIST(root='./others/data', train=True, download=True, transform=transform)
    
    val_size = 5000
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def visualize_gan_results(model: GANModel, n_samples: int = 16):
    """可视化生成结果"""
    samples = model.generate_samples(n_samples)
    samples = samples.cpu().numpy()
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    os.makedirs('./others/results', exist_ok=True)
    save_path = './others/results/gan_results.png'
    plt.savefig(save_path)
    print(f"结果已保存至: {save_path}")
    plt.show()

def main():
    """主函数"""
    print("开始 GAN MNIST 训练...")
    
    # 路径准备
    os.makedirs('./others/results', exist_ok=True)
    os.makedirs('./others/checkpoints', exist_ok=True)
    os.makedirs('./others/logs', exist_ok=True)
    
    # 获取数据
    train_loader, val_loader = get_mnist_dataloaders(batch_size=64)
    print(f"数据加载完成. 训练集批次数: {len(train_loader)}")
    
    # 创建模型
    model = GANModel()
    
    # 准备日志回调
    logging_cb = LoggingCallback(
        log_frequency=100,
        log_dir='./others/logs',
        log_filename='gan_training.log',
        enable_console=False
    )

    # 采样动画回调 - 在每个epoch结束后进行采样并生成动画
    animation_cb = SamplingAnimationCallback(
        n_samples=16,                      # 每次采样16个样本
        save_dir='./others/animations',    # 保存目录
        animation_filename='gan_training.gif',   # 动画文件名
        animation_format='gif',            # 动画格式
        fps=2,                             # 每秒2帧
        sample_method='generate_samples',  # 使用模型的generate_samples方法
        save_intermediate_images=True,     # 保存每个epoch的中间图像
        figsize=(10, 10),                  # 图像大小
        cmap='gray',                       # 灰度图
        show_epoch_label=True,             # 显示epoch标签
        enable_preview=False               # 训练结束后不自动预览
    )
    
    # 初始化 Trainer
    # 注意: 如果有 GPU 请将 device 改为 'cuda' 或 '0'
    trainer = Trainer(
        max_epochs=20,
        device='cpu',
        callbacks=[logging_cb, animation_cb]  # 添加采样动画回调
    )
    
    # 配置 Logger 和检查点
    trainer.setup_logger(
        experiment_name='gan_mnist',
        log_dir='./others/logs',
        checkpoint_dir='./others/checkpoints',
        monitor='val_g_loss',
        mode='min',
        save_top_k=3,
        enable_tensorboard=True,
        enable_total_progress=True
    )
    
    # 开始训练
    print("训练启动中...")
    trainer.fit(model, train_loader, val_loader)
    
    # 生成可视化结果
    print("生成可视化样本...")
    visualize_gan_results(model)
    
    # 保存最终检查点
    final_path = model.save_checkpoint(base_dir='./others/ckpt', naming_keys=['gan_final'])
    print(f"模型已保存至: {final_path}")
    print("训练任务完成!")
    print("采样动画保存在 ./others/animations/ 目录下")

if __name__ == '__main__':
    main()
