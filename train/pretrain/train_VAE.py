import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from xdl.callbacks.logging_callback import LoggingCallback
from xdl.callbacks.sampling_animation_callback import SamplingAnimationCallback

# 导入项目的trainer框架
from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer

# 设置中文字体 (Linux系统使用文泉驿字体)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
VAE MNIST训练脚本
使用项目的trainer框架进行VAE模型的训练
"""

class VAEModel(CoreModel):
    """
    VAE模型类,继承自CoreModel
    包含Encoder和Decoder结构
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 20):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder结构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 均值和方差输出层
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder结构
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # MNIST数据在[0,1]范围内
        )

    def encode(self, x):
        """编码器:输入x -> 输出均值和方差"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码器:从潜在空间z重建x"""
        return self.decoder(z)

    def forward(self, x):
        """前向传播"""
        # 编码
        mu, logvar = self.encode(x.view(-1, self.input_dim))

        # 重参数化
        z = self.reparameterize(mu, logvar)

        # 解码重建
        recon_x = self.decode(z)

        return recon_x, mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar):
        """
        VAE损失函数 = 重构损失 + KL散度 (均值版本)
        """
        # 重构损失 (二元交叉熵) - 使用 mean 得到每个维度的平均损失
        bce = F.binary_cross_entropy(
            recon_x, x.view(-1, self.input_dim), reduction='mean')

        # KL散度损失 - 原始公式是总和,为了与 mean 版本的 BCE 匹配,
        # 需要除以 (batch_size * input_dim)
        batch_size = x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss /= (batch_size * self.input_dim)

        return bce + kld_loss, bce, kld_loss

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        data, _ = batch
        data = data.to(next(self.parameters()).device)
        data = data.view(-1, self.input_dim)

        # 前向传播
        recon_batch, mu, logvar, z = self.forward(data)

        # 计算损失
        total_loss, recon_loss, kld_loss = self.loss_function(
            recon_batch, data, mu, logvar)

        # 记录多个指标
        self.log('loss', total_loss.item())
        self.log('recon', recon_loss.item())
        self.log('kld', kld_loss.item())

        # 计算并记录一些额外指标用于测试日志系统
        recon_ratio = recon_loss.item() / (total_loss.item() + 1e-8)
        kld_ratio = kld_loss.item() / (total_loss.item() + 1e-8)
        self.log('recon_ratio', recon_ratio)
        self.log('kld_ratio', kld_ratio)

        # 反向传播
        self.manual_backward(total_loss)

        # 梯度更新
        for optimizer in self.optimizers:
            optimizer.step()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        data, _ = batch
        # 确保数据在正确的设备上
        data = data.to(next(self.parameters()).device)
        data = data.view(-1, self.input_dim)

        # 前向传播
        recon_batch, mu, logvar, z = self.forward(data)

        # 计算损失
        total_loss, recon_loss, kld_loss = self.loss_function(
            recon_batch, data, mu, logvar)

        # 记录损失 - 使用简化的log方法
        self.log('val_loss', total_loss.item())
        self.log('val_recon_loss', recon_loss.item())
        self.log('val_kld_loss', kld_loss.item())

    def configure_optimizers(self):
        """配置优化器"""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def generate_samples(self, n_samples: int = 16):
        """从潜在空间生成新样本"""
        with torch.no_grad():
            # 从标准正态分布采样潜在变量
            z = torch.randn(n_samples, self.latent_dim)
            # 确保数据在模型参数的设备上
            z = z.to(next(self.parameters()).device)

            # 生成样本
            samples = self.decode(z)
            return samples.view(n_samples, 28, 28)

    def reconstruct_samples(self, data):
        """重建输入样本"""
        with torch.no_grad():
            recon_batch, _, _, _ = self.forward(data.view(-1, self.input_dim))
            return recon_batch.view(-1, 28, 28)


class FlattenTransform:
    """可序列化的数据变换类"""

    def __call__(self, x):
        return x.view(-1)


def get_mnist_dataloaders(batch_size: int = 128, val_split: float = 0.1):
    """获取MNIST数据加载器"""

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        FlattenTransform()  # 展平为784维向量
    ])

    # 下载MNIST数据集
    full_dataset = datasets.MNIST(
        root='./others/data',
        train=True,
        download=True,
        transform=transform
    )

    # 分割训练集和验证集
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    # 测试数据集
    test_dataset = datasets.MNIST(
        root='./others/data',
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器(禁用多进程以避免序列化问题)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def visualize_results(model: VAEModel, test_loader: DataLoader, n_samples: int = 8):
    """可视化训练结果"""

    # 获取测试数据
    data, labels = next(iter(test_loader))

    # 确保数据在正确的设备上
    data = data.to(model.device)

    # 重建样本
    with torch.no_grad():
        original = data[:n_samples].view(n_samples, 28, 28)
        reconstructed = model.reconstruct_samples(data[:n_samples])

        # 生成新样本
        generated = model.generate_samples(n_samples)

    # 创建图像
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))

    for i in range(n_samples):
        # 原始图像
        axes[0, i].imshow(original[i].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'原始 {i+1}')
        axes[0, i].axis('off')

        # 重建图像
        axes[1, i].imshow(reconstructed[i].cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'重建 {i+1}')
        axes[1, i].axis('off')

        # 生成图像
        axes[2, i].imshow(generated[i].cpu().numpy(), cmap='gray')
        axes[2, i].set_title(f'生成 {i+1}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('./others/results/vae_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def visualize_latent_space(model: VAEModel, test_loader: DataLoader, n_points: int = 1000):
    """可视化潜在空间(仅当潜在空间维度为2时)"""

    if model.latent_dim != 2:
        print("潜在空间维度不为2, 跳过可视化")
        return

    # 收集潜在空间表示
    all_z = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            # 确保数据在正确的设备上
            data = data.to(next(model.parameters()).device)
            _, _, _, z = model.forward(data)
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.numpy())

            if len(all_z) * len(z) >= n_points:
                break

    # 合并数据
    z = np.concatenate(all_z)[:n_points]
    labels = np.concatenate(all_labels)[:n_points]

    # 绘制潜在空间
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='数字类别')
    plt.xlabel('潜在维度 1')
    plt.ylabel('潜在维度 2')
    plt.title('VAE潜在空间可视化 (2D)')
    plt.grid(True, alpha=0.3)

    # 添加图例
    legend1 = plt.legend(*scatter.legend_elements(),
                         title="数字类别",
                         loc="upper right")
    plt.gca().add_artist(legend1)

    plt.tight_layout()
    plt.savefig('./others/results/vae_latent_space.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """主训练函数"""

    print("开始VAE MNIST训练...")

    # 创建结果目录
    os.makedirs('./others/results', exist_ok=True)
    os.makedirs('./others/checkpoints', exist_ok=True)

    # 设置训练参数
    batch_size = 128
    max_epochs = 10  # 减少epoch数量用于快速测试
    learning_rate = 1e-3

    print(
        f"训练参数: batch_size={batch_size}, epochs={max_epochs}, lr={learning_rate}")

    # 获取数据
    print("加载MNIST数据集...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size)
    print(f"训练集大小: {len(train_loader)}, 验证集大小: {len(val_loader)}")

    # 创建模型
    print("创建VAE模型...")
    model = VAEModel(input_dim=784, hidden_dim=256,
                     latent_dim=2)  # 使用2维潜在空间便于可视化

    # 准备回调
    log_dir = './others/logs'
    logging_cb = LoggingCallback(
        log_frequency=20,
        log_dir=log_dir,
        log_filename='vae_training_test.log',
        rotation="10 MB",
        enable_console=False
    )

    # 采样动画回调 - 在每个epoch结束后进行采样并生成动画
    animation_cb = SamplingAnimationCallback(
        n_samples=16,                      # 每次采样16个样本
        save_dir='./others/animations',    # 保存目录
        animation_filename='vae_training.gif',  # 动画文件名
        animation_format='gif',            # 动画格式
        fps=2,                             # 每秒2帧
        sample_method='generate_samples',  # 使用模型的generate_samples方法
        save_intermediate_images=True,     # 保存每个epoch的中间图像
        figsize=(10, 10),                  # 图像大小
        cmap='gray',                       # 灰度图
        show_epoch_label=True,             # 显示epoch标签
        enable_preview=False               # 训练结束后不自动预览
    )

    # 创建Trainer
    print("初始化Trainer...")
    trainer = Trainer(
        max_epochs=max_epochs,
        device='cuda',                     # 使用GPU训练
        precision='bf16',                  # 使用bf16精度
        callbacks=[logging_cb, animation_cb]  # 添加采样动画回调
    )

    # 配置日志和检查点
    trainer.setup_logger(
        experiment_name='vae_mnist',
        log_dir=log_dir,
        checkpoint_dir='./others/checkpoints',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        log_every_n_steps=20,
        enable_tqdm=True,
        enable_console=False,
        enable_tensorboard=True,
        enable_total_progress=True,
        tqdm_metric_keys=['loss', 'recon', 'kld', 'lr']
    )

    # 开始训练
    print("开始训练...")
    # 演示:每 100 步作为一个虚拟 epoch 进行验证和日志记录
    trainer.fit(model, train_loader, val_loader, val_check_interval=100)

    # 保存最终模型
    print("保存最终模型...")
    final_model_path = model.save_checkpoint(
        base_dir='./others/ckpt',
        naming_keys=['final'],
        save_optimizer=True
    )
    print(f"模型已保存到: {final_model_path}")

    # 可视化结果
    print("生成可视化结果...")
    model.eval()
    visualize_results(model, test_loader, n_samples=8)

    # 可视化潜在空间(如果维度为2)
    visualize_latent_space(model, test_loader, n_points=1000)

    print("训练完成!")
    print("结果保存在 ./others/results/ 目录下")
    print("检查点保存在 ./others/checkpoints/ 目录下")
    print("日志保存在 ./others/logs/ 目录下")
    print("采样动画保存在 ./others/animations/ 目录下")


if __name__ == '__main__':
    main()
