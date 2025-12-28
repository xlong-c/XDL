import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader

# ====================== 1. 全局配置 ======================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 数据参数
BATCH_SIZE = 128
IMAGE_SIZE = 28  # MNIST图像尺寸
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE  # 784维输入
NUM_SAMPLES_TO_SHOW = 10  # 动画中展示的样本数

# 模型参数
LATENT_DIM = 64  # VAE/GAN隐空间维度
HIDDEN_DIM = 512  # 隐藏层维度
LR = 1e-3         # 学习率
EPOCHS = 30       # 训练轮次
NOISE_DIM = LATENT_DIM  # GAN噪声维度

# 动画参数
ANIMATION_INTERVAL = 300  # 帧间隔（ms）
FIG_SIZE = (12, 6)        # 画布大小

# ====================== 2. 加载MNIST数据集 ======================
# 数据预处理：转为张量 + 归一化到[0,1]（适配sigmoid输出）
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,255] → [0,1]
    transforms.Lambda(lambda x: x.flatten())  # 展平为784维向量
])

# 加载训练集（自动下载）
train_dataset = torchvision.datasets.MNIST(
    root='./mnist_data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 提取一批真实样本用于动画对比
real_samples, _ = next(iter(train_loader))
real_samples = real_samples[:NUM_SAMPLES_TO_SHOW].to(DEVICE)

# ====================== 3. 定义VAE模型（适配MNIST） ======================
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=64):
        super(VAE, self).__init__()
        # 编码器：784 → 512 → 512 → 均值/方差（64维）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器：64 → 512 → 512 → 784（sigmoid输出[0,1]）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出归一化到[0,1]，匹配MNIST像素范围
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE损失：BCE重构损失（适合图像） + KL散度"""
    recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)  # 用BCE替代MSE，更适合二值化图像
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# ====================== 4. 定义GAN模型（适配MNIST） ======================
class Generator(nn.Module):
    """GAN生成器：噪声→784维图像向量（sigmoid输出[0,1]）"""
    def __init__(self, noise_dim=64, hidden_dim=512, output_dim=784):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # 输出像素值[0,1]
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """GAN判别器：784维图像→真假概率（sigmoid输出）"""
    def __init__(self, input_dim=784, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # LeakyReLU避免梯度消失
            nn.Dropout(0.3),    # Dropout正则化
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ====================== 5. 训练模型并记录生成结果 ======================
def train_models():
    """训练VAE和GAN，记录每轮生成的MNIST样本"""
    # 初始化模型
    vae = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    gen = Generator(noise_dim=NOISE_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM).to(DEVICE)
    disc = Discriminator(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)

    # 优化器
    vae_optimizer = optim.Adam(vae.parameters(), lr=LR, betas=(0.9, 0.999))
    gen_optimizer = optim.Adam(gen.parameters(), lr=LR, betas=(0.9, 0.999))
    disc_optimizer = optim.Adam(disc.parameters(), lr=LR, betas=(0.9, 0.999))

    # 损失函数（GAN）
    adversarial_loss = nn.BCELoss()

    # 记录每轮生成的样本（用于动画）
    vae_generated_samples = []
    gan_generated_samples = []

    # 训练循环
    for epoch in range(EPOCHS):
        vae.train()
        gen.train()
        disc.train()
        epoch_vae_loss = 0.0
        epoch_disc_loss = 0.0
        epoch_gen_loss = 0.0

        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # --------------------- 训练VAE ---------------------
            vae_optimizer.zero_grad()
            recon_imgs, mu, logvar = vae(real_imgs)
            loss_vae = vae_loss(recon_imgs, real_imgs, mu, logvar)
            loss_vae.backward()
            vae_optimizer.step()
            epoch_vae_loss += loss_vae.item()

            # --------------------- 训练GAN ---------------------
            # 标签平滑：真实标签=0.9，假标签=0.1（提升稳定性）
            real_labels = torch.full((batch_size, 1), 0.9, device=DEVICE)
            fake_labels = torch.full((batch_size, 1), 0.1, device=DEVICE)

            # 1. 训练判别器
            disc_optimizer.zero_grad()
            # 真实图像损失
            real_pred = disc(real_imgs)
            loss_disc_real = adversarial_loss(real_pred, real_labels)
            # 生成图像损失
            noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
            fake_imgs = gen(noise)
            fake_pred = disc(fake_imgs.detach())
            loss_disc_fake = adversarial_loss(fake_pred, fake_labels)
            # 总判别器损失
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            disc_optimizer.step()
            epoch_disc_loss += loss_disc.item()

            # 2. 训练生成器
            gen_optimizer.zero_grad()
            fake_pred = disc(fake_imgs)
            loss_gen = adversarial_loss(fake_pred, real_labels)
            loss_gen.backward()
            gen_optimizer.step()
            epoch_gen_loss += loss_gen.item()

        # --------------------- 记录本轮生成样本 ---------------------
        vae.eval()
        gen.eval()
        with torch.no_grad():
            # VAE：从隐空间采样生成NUM_SAMPLES_TO_SHOW个样本
            z_vae = torch.randn(NUM_SAMPLES_TO_SHOW, LATENT_DIM, device=DEVICE)
            vae_gen = vae.decode(z_vae).cpu().numpy()
            vae_generated_samples.append(vae_gen)

            # GAN：从噪声生成NUM_SAMPLES_TO_SHOW个样本
            z_gan = torch.randn(NUM_SAMPLES_TO_SHOW, NOISE_DIM, device=DEVICE)
            gan_gen = gen(z_gan).cpu().numpy()
            gan_generated_samples.append(gan_gen)

        # 打印训练进度
        avg_vae_loss = epoch_vae_loss / len(train_loader)
        avg_disc_loss = epoch_disc_loss / len(train_loader)
        avg_gen_loss = epoch_gen_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | VAE Loss: {avg_vae_loss:.2f} | GAN Disc Loss: {avg_disc_loss:.4f} | GAN Gen Loss: {avg_gen_loss:.4f}")

    return vae_generated_samples, gan_generated_samples

# ====================== 6. 制作MNIST生成动画 ======================
def create_mnist_animation(vae_samples, gan_samples):
    """
    动画展示：
    - 第一行：真实MNIST样本
    - 第二行：VAE生成样本（随epoch更新）
    - 第三行：GAN生成样本（随epoch更新）
    """
    # 初始化画布
    fig, axes = plt.subplots(3, NUM_SAMPLES_TO_SHOW, figsize=FIG_SIZE)
    fig.suptitle(f"VAE vs GAN: MNIST Generation (Epoch 1/{EPOCHS})", fontsize=14)

    # 第一行：固定显示真实样本
    real_imgs = real_samples.cpu().numpy()
    for i, ax in enumerate(axes[0]):
        ax.imshow(real_imgs[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title("Real", fontsize=10)

    # 第二行：VAE生成样本（动态更新）
    vae_imgs = [ax.imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE)), cmap='gray') for ax in axes[1]]
    for i, ax in enumerate(axes[1]):
        ax.axis('off')
        if i == 0:
            ax.set_title("VAE Generated", fontsize=10)

    # 第三行：GAN生成样本（动态更新）
    gan_imgs = [ax.imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE)), cmap='gray') for ax in axes[2]]
    for i, ax in enumerate(axes[2]):
        ax.axis('off')
        if i == 0:
            ax.set_title("GAN Generated", fontsize=10)

    # 动画更新函数
    def update(frame):
        # 更新标题（显示当前epoch）
        fig.suptitle(f"VAE vs GAN: MNIST Generation (Epoch {frame+1}/{EPOCHS})", fontsize=14)
        
        # 更新VAE生成图像
        vae_batch = vae_samples[frame]
        for i, img_obj in enumerate(vae_imgs):
            img = vae_batch[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
            img_obj.set_data(img)
        
        # 更新GAN生成图像
        gan_batch = gan_samples[frame]
        for i, img_obj in enumerate(gan_imgs):
            img = gan_batch[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
            img_obj.set_data(img)
        
        return vae_imgs + gan_imgs

    # 创建动画
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=EPOCHS,
        interval=ANIMATION_INTERVAL,
        blit=True,
        repeat=True
    )

    # 显示动画
    plt.tight_layout()
    plt.show()

    # 可选：保存动画（需要ffmpeg）
    # anim.save('mnist_vae_gan_animation.mp4', writer='ffmpeg', fps=3)

# ====================== 7. 主程序入口 ======================
if __name__ == "__main__":
    # 训练模型
    print("Starting training VAE and GAN on MNIST...")
    vae_samples, gan_samples = train_models()
    
    # 生成动画
    print("Creating animation...")
    create_mnist_animation(vae_samples, gan_samples)