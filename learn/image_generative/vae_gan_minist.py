import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
from torch.nn import functional as F
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

# 模型参数（简化配置，与收敛版本一致）
LATENT_DIM = 20  # VAE/GAN隐空间维度（从64降到20）
HIDDEN_DIM = 400  # 隐藏层维度（从512降到400）
LR = 1e-3         # 学习率
EPOCHS = 100      # 训练轮次（从30增加到100）
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
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 提取一批真实样本用于动画对比
real_samples, _ = next(iter(train_loader))
real_samples = real_samples[:NUM_SAMPLES_TO_SHOW].to(DEVICE)

# ====================== 3. 定义VAE模型（适配MNIST，简化版本） ======================
class VAE(nn.Module):
    """简化版VAE，与收敛版本结构一致"""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        # 编码器：784 → 400 → 均值/方差（20维）
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器：20 → 400 → 784（sigmoid输出[0,1]）
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE损失：BCE重构损失（适合图像） + KL散度"""
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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
def train_vae_only():
    """先单独训练VAE，确保收敛"""
    # 初始化VAE模型
    vae = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    vae_optimizer = optim.Adam(vae.parameters(), lr=LR)

    # 记录每轮生成的样本（用于动画）
    vae_generated_samples = []

    # 训练循环
    for epoch in range(EPOCHS):
        vae.train()
        epoch_vae_loss = 0.0

        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(DEVICE)

            # 训练VAE
            vae_optimizer.zero_grad()
            recon_imgs, mu, logvar = vae(real_imgs)
            loss_vae = vae_loss(recon_imgs, real_imgs, mu, logvar)
            loss_vae.backward()
            vae_optimizer.step()
            epoch_vae_loss += loss_vae.item()

        # 记录本轮生成样本
        vae.eval()
        with torch.no_grad():
            z_vae = torch.randn(NUM_SAMPLES_TO_SHOW, LATENT_DIM, device=DEVICE)
            vae_gen = vae.decode(z_vae).cpu().numpy()
            vae_generated_samples.append(vae_gen)

        # 打印训练进度
        avg_vae_loss = epoch_vae_loss / len(train_loader.dataset)  # type: ignore[arg-type]
        print(f"Epoch [{epoch+1}/{EPOCHS}] | VAE Loss: {avg_vae_loss:.4f}")

    return vae_generated_samples

def train_gan_only():
    """单独训练GAN（可选，如果需要对比）"""
    # 初始化GAN模型
    gen = Generator(noise_dim=NOISE_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM).to(DEVICE)
    disc = Discriminator(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)

    # 优化器
    gen_optimizer = optim.Adam(gen.parameters(), lr=LR)
    disc_optimizer = optim.Adam(disc.parameters(), lr=LR)

    # 损失函数
    adversarial_loss = nn.BCELoss()

    # 记录每轮生成的样本
    gan_generated_samples = []

    # 训练循环
    for epoch in range(EPOCHS):
        gen.train()
        disc.train()
        epoch_disc_loss = 0.0
        epoch_gen_loss = 0.0

        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # 标签平滑
            real_labels = torch.full((batch_size, 1), 0.9, device=DEVICE)
            fake_labels = torch.full((batch_size, 1), 0.1, device=DEVICE)

            # 1. 训练判别器
            disc_optimizer.zero_grad()
            real_pred = disc(real_imgs)
            loss_disc_real = adversarial_loss(real_pred, real_labels)

            noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
            fake_imgs = gen(noise)
            fake_pred = disc(fake_imgs.detach())
            loss_disc_fake = adversarial_loss(fake_pred, fake_labels)

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

        # 记录本轮生成样本
        gen.eval()
        with torch.no_grad():
            z_gan = torch.randn(NUM_SAMPLES_TO_SHOW, NOISE_DIM, device=DEVICE)
            gan_gen = gen(z_gan).cpu().numpy()
            gan_generated_samples.append(gan_gen)

        # 打印训练进度
        avg_disc_loss = epoch_disc_loss / len(train_loader)
        avg_gen_loss = epoch_gen_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | GAN Disc Loss: {avg_disc_loss:.4f} | GAN Gen Loss: {avg_gen_loss:.4f}")

    return gan_generated_samples

def train_models():
    """训练VAE（确保收敛）+ GAN（可选对比）"""
    print("=" * 60)
    print("开始训练VAE...")
    print("=" * 60)
    vae_samples = train_vae_only()

    print("\n" + "=" * 60)
    print("开始训练GAN...")
    print("=" * 60)
    gan_samples = train_gan_only()

    return vae_samples, gan_samples

# ====================== 6. 制作MNIST生成动画 ======================
def create_mnist_animation(vae_samples=None, gan_samples=None):
    """
    自适应动画展示：
    - 始终显示：真实MNIST样本（第一行）
    - 可选显示：VAE生成样本（第二行，如果提供）
    - 可选显示：GAN生成样本（第三行，如果提供）
    """
    # 检测需要显示的行数
    show_vae = vae_samples is not None
    show_gan = gan_samples is not None

    if not show_vae and not show_gan:
        print("错误：至少需要提供一种生成样本！")
        return

    # 确定行数和标题
    nrows = 1  # 真实样本（始终显示）
    rows_config = [("Real", "real_samples")]

    if show_vae:
        nrows += 1
        rows_config.append(("VAE Generated", "vae"))
    if show_gan:
        nrows += 1
        rows_config.append(("GAN Generated", "gan"))

    # 初始化画布
    fig, axes = plt.subplots(nrows, NUM_SAMPLES_TO_SHOW, figsize=(FIG_SIZE[0], FIG_SIZE[1] * nrows / 3))

    # 如果只有一行，axes是一维数组，需要统一处理
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # 设置初始标题
    model_names = " vs ".join([name for name, _ in rows_config[1:]])
    fig.suptitle(f"{model_names}: MNIST Generation (Epoch 1/{EPOCHS})", fontsize=14)

    # 存储需要更新的图像对象
    update_targets = []

    # 初始化每一行
    for row_idx, (title, row_type) in enumerate(rows_config):
        if row_type == "real_samples":
            # 第一行：固定显示真实样本
            real_imgs = real_samples.cpu().numpy()
            for i in range(NUM_SAMPLES_TO_SHOW):
                axes[row_idx, i].imshow(real_imgs[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
                axes[row_idx, i].axis('off')
                if i == 0:
                    axes[row_idx, i].set_title(title, fontsize=10)

        elif row_type == "vae":
            # VAE生成样本行（动态更新）
            vae_imgs = []
            for i in range(NUM_SAMPLES_TO_SHOW):
                img_obj = axes[row_idx, i].imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE)), cmap='gray', vmin=0, vmax=1)
                vae_imgs.append(img_obj)
                axes[row_idx, i].axis('off')
                if i == 0:
                    axes[row_idx, i].set_title(title, fontsize=10)
            update_targets.append(("vae", vae_imgs))

        elif row_type == "gan":
            # GAN生成样本行（动态更新）
            gan_imgs = []
            for i in range(NUM_SAMPLES_TO_SHOW):
                img_obj = axes[row_idx, i].imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE)), cmap='gray', vmin=0, vmax=1)
                gan_imgs.append(img_obj)
                axes[row_idx, i].axis('off')
                if i == 0:
                    axes[row_idx, i].set_title(title, fontsize=10)
            update_targets.append(("gan", gan_imgs))

    # 动画更新函数
    def update(frame):
        # 更新标题（显示当前epoch）
        fig.suptitle(f"{model_names}: MNIST Generation (Epoch {frame+1}/{EPOCHS})", fontsize=14)

        updated_imgs = []
        for data_type, imgs in update_targets:
            if data_type == "vae" and vae_samples is not None:
                batch = vae_samples[frame]
                for i, img_obj in enumerate(imgs):
                    img = batch[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
                    img_obj.set_data(img)
                    updated_imgs.append(img_obj)
            elif data_type == "gan" and gan_samples is not None:
                batch = gan_samples[frame]
                for i, img_obj in enumerate(imgs):
                    img = batch[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
                    img_obj.set_data(img)
                    updated_imgs.append(img_obj)

        return updated_imgs

    # 创建动画
    # At least one of vae_samples or gan_samples is not None (checked above)
    frames_source = vae_samples if vae_samples is not None else gan_samples
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(frames_source),  # type: ignore[arg-type]
        interval=ANIMATION_INTERVAL,
        blit=True,
        repeat=True
    )

    # 保存动画为GIF格式（不需要ffmpeg，更通用）
    output_file = f"mnist_{'_vs_'.join([t for _, t in rows_config[1:]])}_animation.gif"
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer='pillow', fps=3)
    print(f"Animation saved successfully to {output_file}!")

    # 显示动画（在某些环境中可能无法正常播放）
    # plt.tight_layout()
    # plt.show()

# ====================== 7. 主程序入口 ======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='VAE vs GAN on MNIST')
    parser.add_argument('--model', type=str, default='vae',
                        choices=['vae', 'gan', 'both'],
                        help='选择训练的模型：vae（仅VAE）, gan（仅GAN）, both（两者都训练）')
    args = parser.parse_args()

    vae_samples = None
    gan_samples = None
    
    if args.model in ['vae', 'both']:
        print("=" * 60)
        print("开始训练VAE...")
        print("=" * 60)
        vae_samples = train_vae_only()

    if args.model in ['gan', 'both']:
        print("\n" + "=" * 60)
        print("开始训练GAN...")
        print("=" * 60)
        gan_samples = train_gan_only()

    # 生成动画（自适应传入的数据）
    print("\nCreating animation...")
    create_mnist_animation(vae_samples=vae_samples, gan_samples=gan_samples)