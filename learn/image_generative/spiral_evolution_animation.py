# -*- coding: utf-8 -*-
"""
生成模型演化动画脚本 (Spiral Evolution Animation)

本脚本展示了多种生成模型(AE, VAE, GAN, DDPM, Flow等)在拟合"双螺旋"数据集过程中的演化过程。
主要特性:
1. 数据生成: 生成带有噪声的双螺旋数据集，并分为 A (蓝色) 和 B (红色) 两部分。
2. 框架集成: 使用 xdl 框架的 Trainer 和 CoreModel 进行模块化开发。
3. 演化记录: 通过 SnapshotCallback 在训练过程中定期捕捉模型生成的点云。
4. 动态可视化: 最终生成 GIF 动画，展示模型从随机分布逐渐收敛到标准双螺旋分布的过程。
5. 着色方案: 生成的点云根据其潜空间分布进行着色，以便观察模型如何将隐变量映射到目标螺旋的分支上。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader, TensorDataset
import os

# 导入 xdl 框架组件
from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer
from xdl.callbacks.base import Callback

# ==========================================
# 1. 数据生成: 增强型双螺旋
# ==========================================


def generate_twin_spiral(n_points=2000, noise=0.02):
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = theta / (4 * np.pi)

    # 螺旋 1
    x1 = r * np.cos(theta) + np.random.randn(n_points) * noise
    y1 = r * np.sin(theta) + np.random.randn(n_points) * noise

    # 螺旋 2
    x2 = -r * np.cos(theta) + np.random.randn(n_points) * noise
    y2 = -r * np.sin(theta) + np.random.randn(n_points) * noise

    data = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    # 标准化数据 (解决拟合问题的关键之一)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.FloatTensor(data)

# ==========================================
# 2. 增强型 MLP 网络
# ==========================================


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU()
        )

    def forward(self, x): return x + self.net(x)


class StrongMLP(nn.Module):
    def __init__(self, in_d, out_d, hidden=128):
        super().__init__()
        self.input_layer = nn.Linear(in_d, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(1)])
        self.output_layer = nn.Linear(hidden, out_d)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)

# ==========================================
# 4. 模型家族 (CoreModel 实现)
# ==========================================


class SnapshotCallback(Callback):
    def __init__(self, model_name, snapshots, fixed_z, frequency=10):
        super().__init__()
        self.model_name, self.snapshots, self.fixed_z, self.frequency = model_name, snapshots, fixed_z, frequency

    def _take_snapshot(self, model):
        model.eval()
        with torch.no_grad():
            points = model.generate_points_from_z(self.fixed_z)
            self.snapshots[self.model_name].append(points.cpu().numpy())
        model.train()

    def on_train_start(self, trainer, model):
        self._take_snapshot(model)

    def on_train_epoch_end(self, trainer, model):
        epoch = trainer.state.current_epoch
        if epoch % self.frequency == 0 or epoch == trainer.max_epochs:
            self._take_snapshot(model)

# --- AE/VAE ---


class AEVAEModel(CoreModel):
    def __init__(self, is_vae=True, lr=2e-4):
        super().__init__()
        self.is_vae = is_vae
        self.lr = lr
        self.enc = StrongMLP(2, 256)
        self.mu = nn.Linear(256, 2)
        self.logvar = nn.Linear(256, 2)
        self.dec = StrongMLP(2, 2)

    def training_step(self, batch, batch_idx):
        x = batch[0].to(self.device)
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * \
            torch.exp(0.5*logvar) if self.is_vae else mu
        recon = self.dec(z)
        loss = F.mse_loss(recon, x)
        if self.is_vae:
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += 0.01 * kld
        self.manual_backward(loss)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

    def generate_points_from_z(self, z): return self.dec(z.to(self.device))
    def configure_optimizers(self): return [
        torch.optim.Adam(self.parameters(), lr=self.lr)]

# --- GAN ---


class GANModel(CoreModel):
    def __init__(self, lr=1e-5):
        super().__init__()
        self.lr = lr
        self.gen = StrongMLP(2, 2)
        self.disc = StrongMLP(2, 1)

    def training_step(self, batch, batch_idx):
        x = batch[0].to(self.device)
        opt_g, opt_d = self.optimizers
        z = torch.randn(x.size(0), 2).to(self.device)
        fake_x = self.gen(z)
        # Train D
        loss_d = F.binary_cross_entropy_with_logits(self.disc(x), torch.ones(x.size(0), 1).to(self.device)) + \
            F.binary_cross_entropy_with_logits(
                self.disc(fake_x.detach()), torch.zeros(x.size(0), 1).to(self.device))
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
        # Train G
        loss_g = F.binary_cross_entropy_with_logits(
            self.disc(fake_x), torch.ones(x.size(0), 1).to(self.device))
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

    def generate_points_from_z(self, z):
        return self.gen(z.to(self.device))

    def configure_optimizers(self):
        return [torch.optim.Adam(
            self.gen.parameters(), lr=self.lr), torch.optim.Adam(self.disc.parameters(), lr=self.lr)]

# --- Iterative ---


class IterativeModel(CoreModel):
    def __init__(self, mode='ddpm', lr=2e-4):
        super().__init__()
        self.mode = mode
        self.lr = lr
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.net = StrongMLP(2 + 64, 2)
        self.steps = 100
        self.beta = torch.linspace(1e-4, 0.02, self.steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward_net(self, x, t):
        t_emb = self.time_embed(t.view(-1, 1))
        return self.net(torch.cat([x, t_emb], dim=1))

    def training_step(self, batch, batch_idx):
        x1 = batch[0].to(self.device)
        t_idx = torch.randint(0, self.steps, (x1.size(0),)).to(self.device)
        t_float = t_idx.float() / self.steps
        if self.mode == 'flow':
            x0 = torch.randn_like(x1)
            xt = (1 - t_float.view(-1, 1)) * x0 + t_float.view(-1, 1) * x1
            target = x1 - x0
        else:
            noise = torch.randn_like(x1)
            a_bar = self.alpha_bar.to(self.device)[t_idx].view(-1, 1)
            xt = torch.sqrt(a_bar) * x1 + torch.sqrt(1 - a_bar) * noise
            target = noise
        pred = self.forward_net(xt, t_float)
        loss = F.mse_loss(pred, target)
        self.manual_backward(loss)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

    def generate_points_from_z(self, z):
        curr_x = z.to(self.device)
        n = z.size(0)
        if self.mode == 'flow':
            steps = 50
            dt = 1.0 / steps
            for i in range(steps):
                curr_x = curr_x + \
                    self.forward_net(curr_x, torch.full(
                        (n, 1), i/float(steps)).to(self.device)) * dt
        elif self.mode == 'ddim':
            steps = 50
            indices = torch.linspace(self.steps-1, 0, steps).long()
            for idx in range(len(indices)):
                i = indices[idx]
                prev_i = indices[idx+1] if idx + \
                    1 < len(indices) else torch.tensor(-1)
                eps = self.forward_net(curr_x, torch.full(
                    (n, 1), i.item()/float(self.steps)).to(self.device))
                a_bar = self.alpha_bar.to(self.device)[i]
                a_bar_prev = self.alpha_bar.to(
                    self.device)[prev_i] if prev_i >= 0 else torch.tensor(1.0).to(self.device)
                pred_x0 = (curr_x - torch.sqrt(1 - a_bar)
                           * eps) / torch.sqrt(a_bar)
                curr_x = torch.sqrt(a_bar_prev) * pred_x0 + \
                    torch.sqrt(1 - a_bar_prev) * eps
        else:
            for i in reversed(range(self.steps)):
                eps = self.forward_net(curr_x, torch.full(
                    (n, 1), i/float(self.steps)).to(self.device))
                a, a_bar = self.alpha.to(self.device)[
                    i], self.alpha_bar.to(self.device)[i]
                curr_x = (1 / torch.sqrt(a)) * (curr_x -
                                                (1-a)/torch.sqrt(1-a_bar) * eps)
                if i > 0:
                    curr_x += torch.sqrt(self.beta.to(self.device)
                                         [i]) * torch.randn_like(curr_x)
        return curr_x

    def configure_optimizers(self): return [
        torch.optim.Adam(self.parameters(), lr=self.lr)]

# ==========================================
# 5. 执行
# ==========================================


def main():
    # ==========================================
    # 配置选项 (在此直接修改参数)
    # ==========================================
    selected_models = ['VAE', 'GAN']  # 可选: AE, VAE, GAN, DDPM, DDIM, Flow
    lr_ae_vae = 1e-5
    lr_gan = 1e-5
    lr_iterative = 2e-4
    epochs = 100
    freq = 5
    # ==========================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = generate_twin_spiral(3000)
    loader = DataLoader(TensorDataset(data), batch_size=128, shuffle=True)

    fixed_z = torch.randn(1000, 2)

    # 初始化所有候选模型工厂
    all_models_factory = {
        'AE': lambda: AEVAEModel(is_vae=False, lr=lr_ae_vae),
        'VAE': lambda: AEVAEModel(is_vae=True, lr=lr_ae_vae),
        'GAN': lambda: GANModel(lr=lr_gan),
        'DDPM': lambda: IterativeModel(mode='ddpm', lr=lr_iterative),
        'DDIM': lambda: IterativeModel(mode='ddim', lr=lr_iterative),
        'Flow': lambda: IterativeModel(mode='flow', lr=lr_iterative)
    }

    # 过滤选定的模型
    selected_names = [n for n in selected_models if n in all_models_factory]
    if not selected_names:
        print(f"未选择任何有效模型。可选: {list(all_models_factory.keys())}")
        return

    models = {name: all_models_factory[name]() for name in selected_names}
    snapshots = {name: [] for name in selected_names}

    for name, model in models.items():
        print(f"正在训练 {name}...")
        trainer = Trainer(max_epochs=epochs, device=device, callbacks=[
                          SnapshotCallback(name, snapshots, fixed_z, frequency=freq)])
        trainer.fit(model, loader)

    # 制作动画
    num_models = len(selected_names)
    cols = min(num_models, 3)
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(
        6 * cols, 6 * rows), squeeze=False)
    axes = axes.flatten()
    scats = []

    # 数据分割用于着色
    n_half_data = data.shape[0] // 2
    n_half_z = fixed_z.shape[0] // 2
    # 前一半 z 映射为蓝色，后一半 z 映射为红色 (模拟双螺旋的两个分支)
    z_colors = ['blue'] * n_half_z + ['red'] * (fixed_z.shape[0] - n_half_z)

    for i, name in enumerate(selected_names):
        ax = axes[i]
        ax.set_title(name, fontsize=15)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        # 绘制背景参考数据：螺旋 a (蓝色), 螺旋 b (红色)
        ax.scatter(data[:n_half_data, 0], data[:n_half_data, 1], s=1, color='blue', alpha=0.05)
        ax.scatter(data[n_half_data:, 0], data[n_half_data:, 1], s=1, color='red', alpha=0.05)
        # 使用零点初始化以匹配 c 参数的长度 (1000)
        scats.append(ax.scatter(np.zeros(len(z_colors)), np.zeros(len(z_colors)), s=4, c=z_colors, alpha=0.8))

    # 隐藏多余的子图
    for j in range(num_models, len(axes)):
        axes[j].axis('off')

    max_frames = max(len(v) for v in snapshots.values())

    def update(frame):
        for i, name in enumerate(selected_names):
            if frame < len(snapshots[name]):
                scats[i].set_offsets(snapshots[name][frame])
        
        # 确定当前显示的 epoch (对应 SnapshotCallback 的采样逻辑)
        if frame == 0:
            current_epoch = 0
        elif frame == max_frames - 1:
            current_epoch = epochs
        else:
            current_epoch = frame * freq
            
        fig.suptitle(
            f"Twin Spiral Fitting Evolution - Epoch: {current_epoch}", fontsize=25)
        return scats

    ani = FuncAnimation(fig, update, frames=max_frames, blit=False)
    os.makedirs('others/results', exist_ok=True)
    save_path = 'others/results/spiral_evolution_fixed.gif'
    print("保存动画中...")
    ani.save(save_path, writer='pillow', fps=10)
    print(f"完成！动画已保存至 {save_path}")


if __name__ == "__main__":
    main()
