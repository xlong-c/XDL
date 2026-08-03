# -*- coding: utf-8 -*-
"""
Double Spiral Training Comparison (GAN vs VAE)
This script trains GAN and VAE on a twin spiral dataset, saves sampling results to CSV,
and generates a comparison animation from the CSV data.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader, TensorDataset

# Import xdl framework components
from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer
from xdl.callbacks.base import Callback

# ==========================================
# 1. Data Generation: Twin Spiral
# ==========================================

def generate_twin_spiral(n_points=2000, noise=0.02):
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = theta / (4 * np.pi)

    # Spiral 1 (Branch A)
    x1 = r * np.cos(theta) + np.random.randn(n_points) * noise
    y1 = r * np.sin(theta) + np.random.randn(n_points) * noise

    # Spiral 2 (Branch B)
    x2 = -r * np.cos(theta) + np.random.randn(n_points) * noise
    y2 = -r * np.sin(theta) + np.random.randn(n_points) * noise

    data1 = np.column_stack([x1, y1])
    data2 = np.column_stack([x2, y2])
    
    # Labeling: 0 for Branch A, 1 for Branch B
    labels1 = np.zeros(n_points)
    labels2 = np.ones(n_points)
    
    data = np.vstack([data1, data2])
    labels = np.concatenate([labels1, labels2])
    
    # Normalize data
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std
    
    return torch.FloatTensor(data), torch.LongTensor(labels), mean, std

# ==========================================
# 2. Model Architecture (Strong MLP with ResBlocks)
# ==========================================

class SimpleMLP(nn.Module):
    def __init__(self, in_d, out_d, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_d)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Model Definitions (CoreModel Implementation)
# ==========================================

class VAEModel(CoreModel):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.enc = SimpleMLP(2, 32)
        self.fc_mu = nn.Linear(32, 2)
        self.fc_logvar = nn.Linear(32, 2)
        self.dec = SimpleMLP(2, 2, hidden=32)

    def training_step(self, batch, batch_idx):
        x = batch[0].to(self.device)
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.dec(z)
        
        recon_loss = F.mse_loss(recon, x)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.05 * kld_loss  # Adjusted KLD weight
        
        self.manual_backward(loss)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()
        self.log('loss', loss.item())

    def generate_points(self, z):
        self.eval()
        with torch.no_grad():
            return self.dec(z.to(self.device))

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)]

class GANModel(CoreModel):
    def __init__(self, lr=5e-5):
        super().__init__()
        self.lr = lr
        self.gen = SimpleMLP(2, 2, hidden=32)
        self.disc = SimpleMLP(2, 1, hidden=32)

    def training_step(self, batch, batch_idx):
        x = batch[0].to(self.device)
        opt_g, opt_d = self.optimizers
        z = torch.randn(x.size(0), 2).to(self.device)
        fake_x = self.gen(z)
        
        # Train Discriminator
        real_pred = self.disc(x)
        fake_pred = self.disc(fake_x.detach())
        loss_d = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) + \
                 F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
        
        # Train Generator
        fake_pred_g = self.disc(fake_x)
        loss_g = F.binary_cross_entropy_with_logits(fake_pred_g, torch.ones_like(fake_pred_g))
        
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()
        
        self.log('g_loss', loss_g.item())
        self.log('d_loss', loss_d.item())

    def generate_points(self, z):
        self.eval()
        with torch.no_grad():
            return self.gen(z.to(self.device))

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.gen.parameters(), lr=self.lr, weight_decay=1e-5),
            torch.optim.Adam(self.disc.parameters(), lr=self.lr, weight_decay=1e-5)
        ]

# ==========================================
# 4. Callbacks for Recording
# ==========================================

class LossHistoryCallback(Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.history = []

    def on_train_epoch_end(self, trainer, model):
        # last_epoch_avg contains the average metrics of the epoch just finished
        avg_metrics = model.last_epoch_avg
        if avg_metrics:
            record = avg_metrics.copy()
            record['epoch'] = trainer.current_epoch
            record['model'] = self.model_name
            self.history.append(record)

class CSVRecorderCallback(Callback):
    def __init__(self, model_name, fixed_z, csv_path, frequency=5):
        super().__init__()
        self.model_name = model_name
        self.fixed_z = fixed_z
        self.csv_path = csv_path
        self.frequency = frequency
        self.records = []

    def _record(self, epoch, model):
        points = model.generate_points(self.fixed_z).cpu().numpy()
        for i, (px, py) in enumerate(points):
            self.records.append({
                'epoch': epoch,
                'model': self.model_name,
                'point_idx': i,
                'x': px,
                'y': py
            })

    def on_train_start(self, trainer, model):
        self._record(0, model)

    def on_train_epoch_end(self, trainer, model):
        epoch = trainer.state.current_epoch
        if epoch % self.frequency == 0 or epoch == trainer.max_epochs:
            self._record(epoch, model)

    def on_train_end(self, trainer, model):
        df = pd.DataFrame(self.records)
        # Append if file exists, else create
        if os.path.exists(self.csv_path):
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_path, index=False)
        self.records = []

# ==========================================
# 5. Visualization Functions
# ==========================================

def plot_loss_curves(vae_history, gan_history, save_path):
    print(f"Plotting loss curves to {save_path}...")
    vae_df = pd.DataFrame(vae_history)
    gan_df = pd.DataFrame(gan_history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # VAE Loss
    if not vae_df.empty:
        ax1.plot(vae_df['epoch'], vae_df['loss'], label='Total Loss')
        ax1.set_title('VAE Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # GAN Loss
    if not gan_df.empty:
        ax2.plot(gan_df['epoch'], gan_df['g_loss'], label='Gen Loss')
        ax2.plot(gan_df['epoch'], gan_df['d_loss'], label='Disc Loss')
        ax2.set_title('GAN Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss curves saved.")

def create_comparison_animation(csv_path, real_data, save_path):
    print(f"Loading data from {csv_path} for animation...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    epochs = sorted(df['epoch'].unique())
    models = sorted(df['model'].unique())
    
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), squeeze=False)
    axes = axes.flatten()
    scats = []
    
    n_half = real_data.shape[0] // 2
    
    # Color coding for generated points based on their index (simulating latent mapping)
    n_points_gen = df[(df['epoch'] == epochs[0]) & (df['model'] == models[0])].shape[0]
    colors_gen = ['blue'] * (n_points_gen // 2) + ['red'] * (n_points_gen - n_points_gen // 2)

    for i, model_name in enumerate(models):
        ax = axes[i]
        ax.set_title(model_name, fontsize=15)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        # Background real data
        ax.scatter(real_data[:n_half, 0], real_data[:n_half, 1], s=1, color='blue', alpha=0.05)
        ax.scatter(real_data[n_half:, 0], real_data[n_half:, 1], s=1, color='red', alpha=0.05)
        # Placeholder for generated points (initialize with zeros matching colors_gen length)
        scat = ax.scatter(np.zeros(len(colors_gen)), np.zeros(len(colors_gen)), s=5, c=colors_gen, alpha=0.6)
        scats.append(scat)

    def update(frame_idx):
        epoch = epochs[frame_idx]
        for i, model_name in enumerate(models):
            data = df[(df['epoch'] == epoch) & (df['model'] == model_name)]
            scats[i].set_offsets(data[['x', 'y']].values)
        fig.suptitle(f"Spiral Fitting Convergence - Epoch: {epoch}", fontsize=20)
        return scats

    ani = FuncAnimation(fig, update, frames=len(epochs), blit=False)
    print("Saving animation...")
    ani.save(save_path, writer='pillow', fps=5)
    print(f"Animation saved to {save_path}")

# ==========================================
# 6. Main Execution
# ==========================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = 'others/results'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'spiral_training_results.csv')
    gif_path = os.path.join(results_dir, 'spiral_vae_gan_comparison.gif')
    loss_plot_path = os.path.join(results_dir, 'spiral_training_losses.png')
    
    # Remove old CSV if exists
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # 1. Prepare Data
    n_points = 2000
    real_data, _, _, _ = generate_twin_spiral(n_points)
    loader = DataLoader(TensorDataset(real_data), batch_size=128, shuffle=True)
    
    # Fixed noise for consistent sampling
    n_samples = 1000
    fixed_z = torch.randn(n_samples, 2)

    # 2. Train VAE
    print("\n--- Training VAE ---")
    vae = VAEModel(lr=1e-4)
    vae_recorder = CSVRecorderCallback('VAE', fixed_z, csv_path, frequency=5)
    vae_loss_cb = LossHistoryCallback('VAE')
    trainer_vae = Trainer(max_epochs=200, device=device, callbacks=[vae_recorder, vae_loss_cb])
    trainer_vae.fit(vae, loader)

    # 3. Train GAN
    print("\n--- Training GAN ---")
    gan = GANModel(lr=5e-5)
    gan_recorder = CSVRecorderCallback('GAN', fixed_z, csv_path, frequency=5)
    gan_loss_cb = LossHistoryCallback('GAN')
    trainer_gan = Trainer(max_epochs=200, device=device, callbacks=[gan_recorder, gan_loss_cb])
    trainer_gan.fit(gan, loader)

    # 4. Plot Losses
    plot_loss_curves(vae_loss_cb.history, gan_loss_cb.history, loss_plot_path)

    # 5. Create Animation from CSV
    create_comparison_animation(csv_path, real_data.numpy(), gif_path)

if __name__ == "__main__":
    main()
