# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置字体, 使用英文避免中文乱码
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class Encoder(nn.Module):
    """改进的Encoder模块, 更适合螺旋数据"""
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3_mean = nn.Linear(hidden_dim//2, latent_dim)
        self.fc3_logvar = nn.Linear(hidden_dim//2, latent_dim)
        self.fc3_ae = nn.Linear(hidden_dim//2, latent_dim)

        # 添加批归一化层以稳定训练
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        z_ae = self.fc3_ae(h)
        z_mean = self.fc3_mean(h)
        z_logvar = self.fc3_logvar(h)
        return z_ae, z_mean, z_logvar

class Decoder(nn.Module):
    """改进的Decoder模块, 更适合螺旋数据"""
    def __init__(self, latent_dim=2, hidden_dim=128, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # 添加批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, z):
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.relu(self.bn2(self.fc2(h)))
        return torch.sigmoid(self.fc3(h))

class VAE(nn.Module):
    """改进的Variational Autoencoder"""
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim=input_dim)

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z_ae, mu, logvar = self.encoder(x)
        reconstructed_ae = self.decoder(z_ae)
        z_vae = self.reparameterize(mu, logvar)
        reconstructed_vae = self.decoder(z_vae)
        return reconstructed_ae, reconstructed_vae, mu, logvar, z_ae, z_vae

def archimedean_spiral_points(n_points=1000, turns=5, noise_level=0.1):
    """
    生成阿基米德螺旋上的点
    :param n_points: 生成点的数量
    :param turns: 螺旋圈数
    :param noise_level: 添加的噪声水平
    :return: 返回螺旋上的点坐标
    """
    # 生成螺旋参数
    theta_max = turns * 2 * np.pi
    theta = np.linspace(0, theta_max, n_points)

    # 阿基米德螺旋: r = a + b*theta
    a = 0.1  # 螺旋的起始半径
    b = 0.5  # 控制螺旋间距的参数
    r = a + b * theta

    # 转换为笛卡尔坐标
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 添加噪声使数据更真实
    noise_x = np.random.normal(0, noise_level, size=x.shape)
    noise_y = np.random.normal(0, noise_level, size=y.shape)
    x += noise_x
    y += noise_y

    return np.column_stack([x, y])

def visualize_latent_space_comparison(vae_model, data, epoch=0, title="Latent Space Visualization"):
    """
    使用阿基米德螺旋可视化VAE和AE的潜在空间
    :param vae_model: 训练的VAE模型
    :param data: 输入数据
    :param epoch: 当前训练轮次
    :param title: 图表标题
    """
    vae_model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data)
        _, _, _, _, z_ae, z_vae = vae_model(data_tensor)

        # 将潜在变量转换为numpy用于绘图
        z_ae_np = z_ae.cpu().numpy()
        z_vae_np = z_vae.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{title} - Epoch {epoch}', fontsize=16)

        # AE的潜在空间可视化
        axes[0].scatter(z_ae_np[:, 0], z_ae_np[:, 1], c='red', alpha=0.6, label='AE Latent Points', s=20)
        axes[0].set_title('Autoencoder Latent Space')
        axes[0].set_xlabel('Latent Dimension 1')
        axes[0].set_ylabel('Latent Dimension 2')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # VAE的潜在空间可视化
        scatter = axes[1].scatter(z_vae_np[:, 0], z_vae_np[:, 1], c=np.sqrt(z_vae_np[:, 0]**2 + z_vae_np[:, 1]**2),
                                 cmap='viridis', alpha=0.6, label='VAE Latent Points', s=20)
        axes[1].set_title('Variational Autoencoder Latent Space')
        axes[1].set_xlabel('Latent Dimension 1')
        axes[1].set_ylabel('Latent Dimension 2')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # 添加颜色条
        plt.colorbar(scatter, ax=axes[1], label='Distance from Origin')

        plt.tight_layout()
        plt.show()

def visualize_archimedean_spiral_training(vae_model, data_generator, epochs=10, save_path=None):
    """
    使用阿基米德螺旋可视化VAE和AE的训练过程对比
    :param vae_model: VAE模型
    :param data_generator: 数据生成函数
    :param epochs: 训练轮数
    :param save_path: 保存动画的路径(可选)
    """
    # 生成阿基米德螺旋数据
    spiral_data = data_generator()
    data_tensor = torch.FloatTensor(spiral_data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('VAE vs AE Training Process - Archimedean Spiral Comparison', fontsize=16)

    # 生成标准阿基米德螺旋用于比较
    true_spiral = archimedean_spiral_points(n_points=500, turns=3, noise_level=0.0)

    # 初始化绘图元素
    ae_scatter = axes[0].scatter([], [], c='red', alpha=0.6, label='AE Latent Space', s=20)
    vae_scatter = axes[1].scatter([], [], c='blue', alpha=0.6, label='VAE Latent Space', s=20)
    true_spiral_plot, = axes[2].plot(true_spiral[:, 0], true_spiral[:, 1], 'g-', alpha=0.7, label='True Spiral', linewidth=2)

    # 设置子图标题
    axes[0].set_title('Autoencoder Latent Space')
    axes[1].set_title('Variational Autoencoder Latent Space')
    axes[2].set_title('True Archimedean Spiral')

    # 设置坐标轴标签
    for ax in axes:
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    def animate(epoch):
        # 在这里模拟训练过程(实际应用中, 这里会调用真实的训练步骤)
        # 为了演示, 我们生成不同的潜在空间分布
        vae_model.train()

        # 这里是模拟更新, 实际实现中这里会包含训练代码
        with torch.no_grad():
            _, _, _, _, z_ae, z_vae = vae_model(data_tensor)
            z_ae_np = z_ae.cpu().numpy()
            z_vae_np = z_vae.cpu().numpy()

        # 清除之前的散点图
        for ax in [axes[0], axes[1]]:
            ax.collections.clear()

        # 重新绘制散点图
        axes[0].scatter(z_ae_np[:, 0], z_ae_np[:, 1], c='red', alpha=0.6, s=20)
        axes[1].scatter(z_vae_np[:, 0], z_vae_np[:, 1], c='blue', alpha=0.6, s=20)

        # 更新标题显示当前epoch
        fig.suptitle(f'VAE vs AE Training Process - Archimedean Spiral Comparison (Epoch {epoch})', fontsize=16)

        return ae_scatter, vae_scatter

    # 创建动画
    ani = animation.FuncAnimation(fig, animate, frames=epochs, interval=1000, blit=False, repeat=True)

    if save_path:
        ani.save(save_path, writer='pillow', fps=1)
    else:
        plt.tight_layout()
        plt.show()

    return ani

def generate_spiral_batch(batch_size=128, turns=3, noise_level=0.1):
    """
    生成阿基米德螺旋批次数据
    :param batch_size: 批次大小
    :param turns: 螺旋圈数
    :param noise_level: 噪声水平
    :return: 螺旋数据点
    """
    return archimedean_spiral_points(n_points=batch_size, turns=turns, noise_level=noise_level)

class SpiralDataGenerator:
    """
    改进的阿基米德螺旋数据生成器, 更好地保持拓扑结构
    """
    def __init__(self, n_points=1000, turns=3, noise_level=0.05, expand_dim=784, scale_to_range=True):
        self.n_points = n_points
        self.turns = turns
        self.noise_level = noise_level
        self.expand_dim = expand_dim
        self.scale_to_range = scale_to_range

    def generate_2d_spiral(self):
        """生成2D阿基米德螺旋数据"""
        theta_max = self.turns * 2 * np.pi
        theta = np.linspace(0, theta_max, self.n_points)

        # 阿基米德螺旋: r = a + b*theta
        a = 0.1  # 螺旋的起始半径
        b = 0.25  # 控制螺旋间距的参数, 调整以适应范围
        r = a + b * theta

        # 转换为笛卡尔坐标
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # 添加噪声使数据更真实
        noise_x = np.random.normal(0, self.noise_level, size=x.shape)
        noise_y = np.random.normal(0, self.noise_level, size=y.shape)
        x += noise_x
        y += noise_y

        # 如果需要, 将数据缩放到 [-1.8, 1.8] 范围内
        if self.scale_to_range:
            max_val = max(np.max(np.abs(x)), np.max(np.abs(y)))
            if max_val > 1.8:
                scale_factor = 1.8 / max_val
                x *= scale_factor
                y *= scale_factor

        return np.column_stack([x, y])

    def generate_expanded_data(self):
        """生成扩展维度的螺旋数据, 使用更智能的映射策略"""
        spiral_2d = self.generate_2d_spiral()

        # 使用更智能的扩展策略
        expanded_data = np.zeros((spiral_2d.shape[0], self.expand_dim))

        # 方法1: 主要坐标重复 + 渐变模式
        repeat_factor = self.expand_dim // 2
        for i in range(spiral_2d.shape[0]):
            x, y = spiral_2d[i]

            # 创建模式：主要坐标重复, 但添加渐变
            pattern_x = np.linspace(x * 0.8, x * 1.2, repeat_factor)
            pattern_y = np.linspace(y * 0.8, y * 1.2, repeat_factor)

            # 交错排列
            expanded_data[i, ::2] = pattern_x
            expanded_data[i, 1::2] = pattern_y[:len(pattern_x)]

        # 如果有奇数维度, 用x的第一个值填充
        if self.expand_dim % 2 != 0:
            expanded_data[i, -1] = spiral_2d[i, 0]

        # 添加少量保持结构的噪声
        expanded_data += np.random.normal(0, 0.005, expanded_data.shape)

        return spiral_2d, expanded_data

    def generate_with_preserved_structure(self):
        """生成具有更好结构保持性的数据"""
        spiral_2d = self.generate_2d_spiral()

        # 使用PCA-like扩展方法
        expanded_data = self._pca_expansion(spiral_2d, self.expand_dim)

        return spiral_2d, expanded_data

    def _pca_expansion(self, data_2d, target_dim):
        """类似PCA的扩展方法, 保持主要结构"""
        n_samples = data_2d.shape[0]

        # 计算数据的协方差矩阵
        centered_data = data_2d - np.mean(data_2d, axis=0)
        cov_matrix = np.cov(centered_data.T)

        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 生成扩展数据
        expanded_data = np.zeros((n_samples, target_dim))

        # 前两个维度使用原始数据
        expanded_data[:, :2] = data_2d

        # 其余维度使用原始数据的线性组合和少量噪声
        for i in range(2, target_dim):
            # 使用前两个主成分的随机组合
            weight1 = np.random.uniform(0.1, 0.3)
            weight2 = np.random.uniform(0.1, 0.3)
            noise = np.random.normal(0, 0.02, n_samples)

            expanded_data[:, i] = (weight1 * data_2d[:, 0] +
                                  weight2 * data_2d[:, 1] +
                                  noise)

        return expanded_data


def improved_spiral_loss(latent_codes, alpha=1.0, target_turns=3, max_radius=2.0):
    """
    改进的螺旋结构损失函数, 更好地约束潜在空间形成阿基米德螺旋
    :param latent_codes: 潜在代码 (batch_size, 2)
    :param alpha: 损失权重
    :param target_turns: 目标螺旋圈数
    :param max_radius: 最大半径
    :return: 螺旋损失
    """
    x = latent_codes[:, 0]
    y = latent_codes[:, 1]

    # 计算极坐标
    r = torch.sqrt(x**2 + y**2 + 1e-8)  # 添加小值避免数值问题
    theta = torch.atan2(y, x)
    theta = torch.where(theta < 0, theta + 2*np.pi, theta)

    # 按角度排序
    sorted_indices = torch.argsort(theta)
    sorted_r = r[sorted_indices]
    sorted_theta = theta[sorted_indices]

    # 阿基米德螺旋的理想关系: r = a + b*theta
    # 其中 a 是起始半径, b = (max_radius - a) / (target_turns * 2π)
    a = 0.1  # 起始半径
    b = (max_radius - a) / (target_turns * 2 * np.pi)
    ideal_r = a + b * sorted_theta

    # 损失1: 拟合理想螺旋形状
    spiral_shape_loss = torch.mean((sorted_r - ideal_r) ** 2)

    # 损失2: 确保角度覆盖范围足够(形成完整的螺旋)
    theta_range = torch.max(sorted_theta) - torch.min(sorted_theta)
    coverage_loss = torch.relu(target_turns * 2 * np.pi - theta_range) ** 2

    # 损失3: 局部单调性(半径应该随着角度大致递增)
    r_diff = sorted_r[1:] - sorted_r[:-1]
    monotonicity_loss = torch.mean(torch.relu(-r_diff) ** 2)

    # 损失4: 角度分布均匀性(避免点聚集)
    theta_diff = sorted_theta[1:] - sorted_theta[:-1]
    # 处理跨越2π的情况
    theta_diff = torch.where(theta_diff < 0, theta_diff + 2*np.pi, theta_diff)
    expected_theta_diff = (theta_range) / (len(theta_diff))
    uniformity_loss = torch.mean((theta_diff - expected_theta_diff) ** 2)

    total_loss = spiral_shape_loss + 0.5 * coverage_loss + 0.3 * monotonicity_loss + 0.2 * uniformity_loss

    return alpha * total_loss


def correlation_loss(latent_codes, target_coords):
    """
    保持潜在代码与原始螺旋坐标的相关性
    :param latent_codes: 潜在代码 (batch_size, 2)
    :param target_coords: 目标坐标 (batch_size, 2)
    :return: 相关性损失
    """
    # 计算皮尔逊相关系数
    x_corr = torch.corrcoef(torch.stack([latent_codes[:, 0], target_coords[:, 0]]))[0, 1]
    y_corr = torch.corrcoef(torch.stack([latent_codes[:, 1], target_coords[:, 1]]))[0, 1]

    # 相关系数越接近1越好
    correlation_loss = (1 - x_corr) ** 2 + (1 - y_corr) ** 2

    return correlation_loss


def improved_train_on_spiral_data(model, data, original_spiral_2d, epochs=200, learning_rate=5e-4,
                                  log_interval=10, spiral_weight=2.0, correlation_weight=0.5):
    """
    改进的螺旋数据训练函数
    :param model: VAE模型
    :param data: 训练数据(高维)
    :param original_spiral_2d: 原始2D螺旋坐标(用于相关性约束)
    :param epochs: 训练轮数
    :param learning_rate: 学习率
    :param log_interval: 日志记录间隔
    :param spiral_weight: 螺旋损失权重
    :param correlation_weight: 相关性损失权重
    :return: 训练历史
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # 用于记录训练历史
    ae_losses = []
    vae_losses = []
    total_losses = []
    recon_ae_losses = []
    recon_vae_losses = []
    kl_losses = []
    spiral_losses = []
    correlation_losses = []

    data_tensor = torch.FloatTensor(data)
    original_tensor = torch.FloatTensor(original_spiral_2d)

    # 计算原始螺旋的参数, 用于损失函数
    target_turns = 3
    max_radius = np.max(np.sqrt(original_spiral_2d[:, 0]**2 + original_spiral_2d[:, 1]**2))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        reconstructed_ae, reconstructed_vae, mu, logvar, z_ae, z_vae = model(data_tensor)

        # 计算各个损失分量
        # AE重构损失
        ae_recon_loss = F.mse_loss(reconstructed_ae, data_tensor)

        # VAE重构损失
        vae_recon_loss = F.mse_loss(reconstructed_vae, data_tensor)

        # VAE KL散度损失
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 改进的螺旋结构损失
        spiral_loss_ae = improved_spiral_loss(z_ae, alpha=1.0, target_turns=target_turns, max_radius=max_radius)
        spiral_loss_vae = improved_spiral_loss(z_vae, alpha=1.0, target_turns=target_turns, max_radius=max_radius)
        total_spiral_loss = spiral_loss_ae + spiral_loss_vae

        # 相关性损失 - 保持潜在代码与原始坐标的相关性
        correlation_loss_ae = correlation_loss(z_ae, original_tensor)
        correlation_loss_vae = correlation_loss(z_vae, original_tensor)
        total_correlation_loss = correlation_loss_ae + correlation_loss_vae

        # 总损失 - 调整权重
        total_loss = (ae_recon_loss +
                     vae_recon_loss +
                     0.001 * kl_divergence +  # 降低KL损失的权重
                     spiral_weight * total_spiral_loss +
                     correlation_weight * total_correlation_loss)

        # 反向传播
        total_loss.backward()

        # 梯度裁剪以稳定训练
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        # 更新学习率
        scheduler.step(total_loss)

        # 记录损失值
        ae_losses.append(ae_recon_loss.item())
        vae_losses.append(vae_recon_loss.item() + kl_divergence.item())
        total_losses.append(total_loss.item())
        recon_ae_losses.append(ae_recon_loss.item())
        recon_vae_losses.append(vae_recon_loss.item())
        kl_losses.append(kl_divergence.item())
        spiral_losses.append(total_spiral_loss.item())
        correlation_losses.append(total_correlation_loss.item())

        if epoch % log_interval == 0:
            print(f'Epoch [{epoch}/{epochs}], '
                  f'Total Loss: {total_loss.item():.6f}, '
                  f'AE Recon: {ae_recon_loss.item():.6f}, '
                  f'VAE Recon: {vae_recon_loss.item():.6f}, '
                  f'KL: {kl_divergence.item():.6f}, '
                  f'Spiral: {total_spiral_loss.item():.6f}, '
                  f'Corr: {total_correlation_loss.item():.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

    return {
        'ae_losses': ae_losses,
        'vae_losses': vae_losses,
        'total_losses': total_losses,
        'recon_ae_losses': recon_ae_losses,
        'recon_vae_losses': recon_vae_losses,
        'kl_losses': kl_losses,
        'spiral_losses': spiral_losses,
        'correlation_losses': correlation_losses
    }


def plot_loss_changes(history, title="Loss Changes During Training"):
    """
    绘制训练过程中各种损失的变化
    :param history: 训练历史字典
    :param title: 图表标题
    """
    # 检查是否有螺旋损失
    has_spiral_loss = 'spiral_losses' in history

    if has_spiral_loss:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    fig.suptitle(title, fontsize=16)

    epochs = range(len(history['total_losses']))

    # 总损失
    axes[0, 0].plot(epochs, history['total_losses'], label='Total Loss', color='purple')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # AE重构损失
    axes[0, 1].plot(epochs, history['recon_ae_losses'], label='AE Reconstruction Loss', color='red')
    axes[0, 1].set_title('AE Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # VAE重构损失
    axes[0, 2].plot(epochs, history['recon_vae_losses'], label='VAE Reconstruction Loss', color='blue')
    axes[0, 2].set_title('VAE Reconstruction Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    # KL散度损失
    axes[1, 0].plot(epochs, history['kl_losses'], label='KL Divergence Loss', color='green')
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    if has_spiral_loss:
        # 螺旋损失
        axes[1, 1].plot(epochs, history['spiral_losses'], label='Spiral Loss', color='orange')
        axes[1, 1].set_title('Spiral Structure Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        # AE总损失
        axes[1, 2].plot(epochs, history['ae_losses'], label='AE Total Loss', color='red')
        axes[1, 2].set_title('AE Total Loss (Recon)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
    else:
        # AE总损失
        axes[1, 1].plot(epochs, history['ae_losses'], label='AE Total Loss', color='red')
        axes[1, 1].set_title('AE Total Loss (Recon)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        # VAE总损失
        axes[1, 2].plot(epochs, history['vae_losses'], label='VAE Total Loss (Recon + KL)', color='blue')
        axes[1, 2].set_title('VAE Total Loss (Recon + KL)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()

    plt.tight_layout()
    plt.show()


def animate_training_process(model, data_generator, epochs=50, save_path=None):
    """
    动画显示VAE和AE的训练过程
    :param model: VAE模型
    :param data_generator: 数据生成器
    :param epochs: 训练轮数
    :param save_path: 保存动画的路径(可选)
    """
    # 获取数据
    _, expanded_data = data_generator.generate_expanded_data()
    data_tensor = torch.FloatTensor(expanded_data)

    # 获取真实的2D螺旋数据用于比较
    true_spiral_2d, _ = data_generator.generate_expanded_data()

    # 设置图形
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 4, figure=fig)

    # 定义子图
    ax_true = fig.add_subplot(gs[0, 0])  # 真实螺旋
    ax_ae_latent = fig.add_subplot(gs[0, 1])  # AE潜在空间
    ax_vae_latent = fig.add_subplot(gs[0, 2])  # VAE潜在空间
    ax_ae_recon = fig.add_subplot(gs[0, 3])  # AE重构
    ax_orig = fig.add_subplot(gs[1, :2])  # 原始高维数据投影
    ax_combined = fig.add_subplot(gs[1, 2:])  # 潜在空间合并视图
    ax_losses = fig.add_subplot(gs[2, :])  # 损失变化

    # 设置坐标轴范围为 -2 到 2
    for ax in [ax_true, ax_ae_latent, ax_vae_latent, ax_ae_recon, ax_orig, ax_combined]:
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-2.1, 2.1)

    # 初始化绘图元素
    true_line, = ax_true.plot(true_spiral_2d[:, 0], true_spiral_2d[:, 1], 'g-', alpha=0.7, label='True Spiral', linewidth=2)
    ae_latent_scatter = ax_ae_latent.scatter([], [], c='red', alpha=0.6, label='AE Latent', s=20)
    vae_latent_scatter = ax_vae_latent.scatter([], [], c='blue', alpha=0.6, label='VAE Latent', s=20)
    ae_recon_scatter = ax_ae_recon.scatter([], [], c='red', alpha=0.6, label='AE Recon', s=20)
    orig_scatter = ax_orig.scatter([], [], c='gray', alpha=0.6, label='Original (Projected)', s=10)
    combined_scatter = ax_combined.scatter([], [], c='red', marker='x', alpha=0.6, label='AE Latent', s=30)
    combined_vae_scatter = ax_combined.scatter([], [], c='blue', marker='o', alpha=0.6, label='VAE Latent', s=20)

    # 设置子图标题
    ax_true.set_title('True Archimedean Spiral')
    ax_ae_latent.set_title('AE Latent Space')
    ax_vae_latent.set_title('VAE Latent Space')
    ax_ae_recon.set_title('AE Reconstruction (Projected)')
    ax_orig.set_title('Original High-Dim Data (PCA-like Projection)')
    ax_combined.set_title('Combined Latent Spaces Comparison')
    ax_losses.set_title('Loss Changes During Training')

    # 设置坐标轴标签
    for ax in [ax_true, ax_ae_latent, ax_vae_latent, ax_ae_recon, ax_orig, ax_combined]:
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)

    # 初始化损失图
    loss_line_total, = ax_losses.plot([], [], label='Total Loss', color='purple')
    loss_line_ae, = ax_losses.plot([], [], label='AE Loss', color='red')
    loss_line_vae, = ax_losses.plot([], [], label='VAE Loss', color='blue')
    ax_losses.set_xlabel('Epoch')
    ax_losses.set_ylabel('Loss')
    ax_losses.legend()
    ax_losses.grid(True, alpha=0.3)

    # 用于存储损失值
    epoch_list = []
    total_losses = []
    ae_losses = []
    vae_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def animate(epoch):
        nonlocal epoch_list, total_losses, ae_losses, vae_losses

        model.train()
        optimizer.zero_grad()

        # 前向传播
        reconstructed_ae, reconstructed_vae, mu, logvar, z_ae, z_vae = model(data_tensor)

        # 计算损失
        ae_recon_loss = F.mse_loss(reconstructed_ae, data_tensor)
        vae_recon_loss = F.mse_loss(reconstructed_vae, data_tensor)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = ae_recon_loss + vae_recon_loss + kl_divergence

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 获取潜在空间表示
        with torch.no_grad():
            z_ae_np = z_ae.cpu().numpy()
            z_vae_np = z_vae.cpu().numpy()

            # 确保数据在 [-2, 2] 范围内
            ae_proj = np.clip(z_ae_np, -2, 2)
            vae_proj = np.clip(z_vae_np, -2, 2)

            # 为了可视化重构结果, 也做类似的简单投影并限制范围
            ae_recon_proj = np.clip(reconstructed_ae[:, :2].cpu().numpy(), -2, 2)

            # 原始数据的简单投影并限制范围
            orig_proj = np.clip(data_tensor[:, :2].cpu().numpy(), -2, 2)

        # 更新潜在空间散点图
        ae_latent_scatter.set_offsets(ae_proj)
        vae_latent_scatter.set_offsets(vae_proj)
        ae_recon_scatter.set_offsets(ae_recon_proj)
        orig_scatter.set_offsets(orig_proj)
        combined_scatter.set_offsets(ae_proj)
        combined_vae_scatter.set_offsets(vae_proj)

        # 更新颜色
        ae_latent_scatter.set_color('red')
        vae_latent_scatter.set_color('blue')
        ae_recon_scatter.set_color('red')
        orig_scatter.set_color('gray')
        combined_scatter.set_color('red')
        combined_vae_scatter.set_color('blue')

        # 记录损失
        epoch_list.append(epoch)
        total_losses.append(total_loss.item())
        ae_losses.append(ae_recon_loss.item())
        vae_losses.append(vae_recon_loss.item() + kl_divergence.item())

        # 更新损失图
        loss_line_total.set_data(epoch_list, total_losses)
        loss_line_ae.set_data(epoch_list, ae_losses)
        loss_line_vae.set_data(epoch_list, vae_losses)

        # 调整损失图的范围
        ax_losses.relim()
        ax_losses.autoscale_view()

        # 更新标题显示当前epoch
        fig.suptitle(f'Training Animation: VAE vs AE on Archimedean Spiral (Epoch {epoch})', fontsize=16)

        return (ae_latent_scatter, vae_latent_scatter, ae_recon_scatter, orig_scatter,
                combined_scatter, combined_vae_scatter, loss_line_total, loss_line_ae, loss_line_vae)

    # 创建动画
    ani = animation.FuncAnimation(fig, animate, frames=epochs, interval=200, blit=False, repeat=True)

    if save_path:
        ani.save(save_path, writer='pillow', fps=5)
    else:
        plt.tight_layout()
        plt.show()

    return ani

def compute_spiral_metrics(latent_codes, true_spiral):
    """
    计算潜在代码与真实螺旋的匹配度量
    :param latent_codes: 潜在代码 (n, 2)
    :param true_spiral: 真实螺旋数据 (n, 2)
    :return: 度量字典
    """
    # 计算重构误差
    recon_error = np.mean(np.sqrt(np.sum((latent_codes - true_spiral) ** 2, axis=1)))

    # 计算螺旋度 - 使用极坐标排序的一致性
    x, y = latent_codes[:, 0], latent_codes[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2*np.pi, theta)  # 标准化到[0, 2π]

    # 检查r和theta之间的单调关系
    sorted_indices = np.argsort(theta)
    sorted_r = r[sorted_indices]

    # 计算半径的单调递增性(螺旋应该大致单调递增)
    r_diff = np.diff(sorted_r)
    monotonicity = np.mean(r_diff > 0)  # 比例正值差分

    return {
        'recon_error': recon_error,
        'monotonicity': monotonicity
    }


def plot_fitting_progress(model, data_generator, epochs_list=[0, 10, 20, 30, 40, 50]):
    """
    可视化训练过程中AE和VAE对螺旋数据的拟合进展
    :param model: VAE模型
    :param data_generator: 数据生成器
    :param epochs_list: 要显示的训练轮次列表
    """
    # 生成数据
    _, expanded_data = data_generator.generate_expanded_data()
    data_tensor = torch.FloatTensor(expanded_data)

    # 获取真实的2D螺旋数据用于比较
    true_spiral_2d, _ = data_generator.generate_expanded_data()

    # 为每个epoch绘制子图
    n_epochs = len(epochs_list)
    fig, axes = plt.subplots(3, n_epochs, figsize=(4*n_epochs, 12))
    if n_epochs == 1:
        axes = axes.reshape(3, 1)

    # 设置坐标轴范围
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(-2.1, 2.1)
            ax.set_ylim(-2.1, 2.1)
            ax.grid(True, alpha=0.3)

    # 保存模型原始状态
    original_state = model.state_dict()

    for idx, epoch in enumerate(epochs_list):
        # 临时训练到指定epoch
        temp_model = VAE(input_dim=784, hidden_dim=128, latent_dim=2)
        temp_model.load_state_dict(original_state)
        temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=1e-2)

        temp_model.train()
        for e in range(epoch):
            temp_optimizer.zero_grad()
            reconstructed_ae, reconstructed_vae, mu, logvar, z_ae, z_vae = temp_model(data_tensor)

            ae_recon_loss = F.mse_loss(reconstructed_ae, data_tensor)
            vae_recon_loss = F.mse_loss(reconstructed_vae, data_tensor)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # 使用改进的螺旋损失函数
            spiral_loss_val = improved_spiral_loss(z_ae) + improved_spiral_loss(z_vae)
            total_loss = ae_recon_loss + vae_recon_loss + kl_divergence + 0.1 * spiral_loss_val

            total_loss.backward()
            temp_optimizer.step()

        # 获取当前epoch的潜在表示
        temp_model.eval()
        with torch.no_grad():
            _, _, _, _, z_ae, z_vae = temp_model(data_tensor)
            z_ae_np = z_ae.cpu().numpy()
            z_vae_np = z_vae.cpu().numpy()

            # 限制数据范围
            ae_proj = np.clip(z_ae_np, -2, 2)
            vae_proj = np.clip(z_vae_np, -2, 2)

        # 绘制AE潜在空间
        axes[0, idx].plot(true_spiral_2d[:, 0], true_spiral_2d[:, 1], 'g-', alpha=0.5, label='True Spiral', linewidth=2)
        scatter_ae = axes[0, idx].scatter(ae_proj[:, 0], ae_proj[:, 1], c=np.arctan2(ae_proj[:, 1], ae_proj[:, 0]),
                                         cmap='hsv', alpha=0.6, s=10, label='AE Fitting')
        axes[0, idx].set_title(f'AE at Epoch {epoch}')
        axes[0, idx].legend()

        # 绘制VAE潜在空间
        axes[1, idx].plot(true_spiral_2d[:, 0], true_spiral_2d[:, 1], 'g-', alpha=0.5, label='True Spiral', linewidth=2)
        scatter_vae = axes[1, idx].scatter(vae_proj[:, 0], vae_proj[:, 1], c=np.arctan2(vae_proj[:, 1], vae_proj[:, 0]),
                                          cmap='hsv', alpha=0.6, s=10, label='VAE Fitting')
        axes[1, idx].set_title(f'VAE at Epoch {epoch}')
        axes[1, idx].legend()

        # 计算并绘制度量
        ae_metrics = compute_spiral_metrics(ae_proj, true_spiral_2d)
        vae_metrics = compute_spiral_metrics(vae_proj, true_spiral_2d)

        # 在第三个子图中显示度量
        axes[2, idx].text(0.1, 0.8, f'AE Monotonicity: {ae_metrics["monotonicity"]:.3f}',
                          transform=axes[2, idx].transAxes, fontsize=10, verticalalignment='top')
        axes[2, idx].text(0.1, 0.6, f'AE Recon Error: {ae_metrics["recon_error"]:.3f}',
                          transform=axes[2, idx].transAxes, fontsize=10, verticalalignment='top')
        axes[2, idx].text(0.1, 0.4, f'VAE Monotonicity: {vae_metrics["monotonicity"]:.3f}',
                          transform=axes[2, idx].transAxes, fontsize=10, verticalalignment='top')
        axes[2, idx].text(0.1, 0.2, f'VAE Recon Error: {vae_metrics["recon_error"]:.3f}',
                          transform=axes[2, idx].transAxes, fontsize=10, verticalalignment='top')
        axes[2, idx].set_xlim(0, 1)
        axes[2, idx].set_ylim(0, 1)
        axes[2, idx].set_title(f'Metrics at Epoch {epoch}')
        axes[2, idx].axis('off')  # 隐藏坐标轴

    plt.suptitle('Fitting Progress: AE vs VAE on Archimedean Spiral', fontsize=16)
    plt.tight_layout()
    plt.show()

# 示例使用代码
if __name__ == "__main__":
    print("Testing improved VAE spiral fitting...")

    # 设置随机种子确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建螺旋数据生成器实例
    print("Creating improved spiral data generator...")
    data_gen = SpiralDataGenerator(n_points=500, turns=2, noise_level=0.05, expand_dim=784)

    # 生成具有更好结构保持性的数据
    print("Generating 2D Archimedean spiral data and expanding to high-dimensional space...")
    spiral_2d, expanded_data = data_gen.generate_with_preserved_structure()

    # 验证数据范围
    print(f"Data range - X: [{spiral_2d[:, 0].min():.3f}, {spiral_2d[:, 0].max():.3f}], "
          f"Y: [{spiral_2d[:, 1].min():.3f}, {spiral_2d[:, 1].max():.3f}]")
    print(f"Spiral parameters: turns={data_gen.turns}, points={data_gen.n_points}")

    # 创建改进的VAE模型 (使用适中的维度)
    print("Creating improved VAE model...")
    vae_model = VAE(input_dim=784, hidden_dim=128, latent_dim=2)

    # 可视化初始状态
    print("Visualizing initial model state...")
    visualize_latent_space_comparison(vae_model, expanded_data, epoch=0,
                                    title="Initial State: Improved VAE vs AE Latent Space")

    # 使用改进的训练方法训练模型
    print("Training VAE model with improved spiral-aware loss...")
    history = improved_train_on_spiral_data(
        vae_model,
        expanded_data,
        spiral_2d,
        epochs=100,
        learning_rate=1e-3,  # 稍微降低学习率以适应更长的训练
        log_interval=10,
        spiral_weight=1.5,   # 增加螺旋损失权重
        correlation_weight=0.8  # 增加相关性损失权重
    )

    # 绘制损失变化
    print("Plotting loss changes during training...")
    plot_loss_changes(history, title="Improved VAE vs AE Loss Changes on Spiral Data")

    # 可视化拟合过程
    print("Visualizing fitting progress...")
    plot_fitting_progress(vae_model, data_gen, epochs_list=[0, 20, 50, 80, 100])

    # 创建最终的对比可视化
    print("Creating final comparison visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Improved VAE Spiral Fitting Results', fontsize=16)

    vae_model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(expanded_data)
        reconstructed_ae, reconstructed_vae, mu, logvar, z_ae, z_vae = vae_model(data_tensor)

        z_ae_np = z_ae.cpu().numpy()
        z_vae_np = z_vae.cpu().numpy()

        # 绘制真实螺旋
        axes[0].plot(spiral_2d[:, 0], spiral_2d[:, 1], 'g-', linewidth=2, alpha=0.7, label='True Spiral')
        axes[0].set_title('True Archimedean Spiral')
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(-2, 2)
        axes[0].set_ylim(-2, 2)

        # VAE潜在空间
        scatter_vae = axes[1].scatter(z_vae_np[:, 0], z_vae_np[:, 1],
                                    c=np.sqrt(z_vae_np[:, 0]**2 + z_vae_np[:, 1]**2),
                                    cmap='viridis', s=20, alpha=0.7)
        axes[1].set_title('VAE Latent Space')
        axes[1].set_xlabel('Latent Dim 1')
        axes[1].set_ylabel('Latent Dim 2')
        plt.colorbar(scatter_vae, ax=axes[1], label='Radius')
        axes[1].set_xlim(-2, 2)
        axes[1].set_ylim(-2, 2)

        # 对比图：VAE vs 真实
        axes[2].plot(spiral_2d[:, 0], spiral_2d[:, 1], 'g-', linewidth=2, alpha=0.5, label='True Spiral')
        axes[2].scatter(z_vae_np[:, 0], z_vae_np[:, 1], c='blue', s=15, alpha=0.6, label='VAE Latent')
        axes[2].set_title('VAE vs True Spiral Comparison')
        axes[2].set_xlabel('X/Latent Dim 1')
        axes[2].set_ylabel('Y/Latent Dim 2')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(-2, 2)
        axes[2].set_ylim(-2, 2)

    plt.tight_layout()
    plt.show()

    # 评估结果
    ae_metrics = compute_spiral_metrics(z_ae_np, spiral_2d)
    vae_metrics = compute_spiral_metrics(z_vae_np, spiral_2d)

    print("\nFinal Results:")
    print(f"AE Monotonicity: {ae_metrics['monotonicity']:.3f}")
    print(f"AE Reconstruction Error: {ae_metrics['recon_error']:.3f}")
    print(f"VAE Monotonicity: {vae_metrics['monotonicity']:.3f}")
    print(f"VAE Reconstruction Error: {vae_metrics['recon_error']:.3f}")

    print("\nImproved VAE spiral fitting example completed.")
    print("Key improvements:")
    print("1. Multi-constraint spiral loss function (shape + coverage + monotonicity + uniformity)")
    print("2. Correlation loss maintaining relationship between latent space and original coordinates")
    print("3. More intelligent high-dimensional expansion strategy")
    print("4. Improved training strategy (learning rate scheduling + weight adjustment)")
