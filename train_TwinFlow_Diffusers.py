"""
TwinFlow 训练器 - 用于 Diffusers 模型的单步/少步生成加速
支持从 Diffusers 加载预训练模型并使用 TwinFlow 进行蒸馏训练

使用示例:
    # 从 diffusers 加载 SD 模型并使用 TwinFlow 训练
    python train_TwinFlow_Diffusers.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1 \
        --dataset_name lambdalabs/pokemon-blip-captions \
        --resolution 512 \
        --batch_size 4 \
        --max_epochs 100

参考:
- TwinFlow论文: https://arxiv.org/abs/2512.05150
- Diffusers: https://github.com/huggingface/diffusers
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image
from tqdm import tqdm

# 导入 xdl 训练框架
from xdl.trainer.coreModel import CoreModel
from xdl.trainer.trainer import Trainer
from xdl.callbacks.logging_callback import LoggingCallback
from xdl.callbacks.tensorboard_callback import TensorBoardCallback
from xdl.callbacks.model_checkpoint import ModelCheckpoint

# 导入 TwinFlow 训练模块
from xdl.model.generate.twinflow import TwinFlow as TwinFlowTrainer 


try:
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        UNet2DConditionModel,
        StableDiffusionPipeline,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0
    from transformers import CLIPTextModel, CLIPTokenizer
    from datasets import load_dataset
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not installed. Install with: pip install diffusers transformers datasets")


class DiffusersUNetWrapper(nn.Module):
    """
    包装 Diffusers 的 UNet 模型，使其兼容 TwinFlow 训练
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        noise_scheduler: DDPMScheduler,
        use_vae_encoding: bool = True,
        resolution: int = 512,
    ):
        super().__init__()
        
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        
        self.use_vae_encoding = use_vae_encoding
        self.resolution = resolution
        
        # VAE 缩放因子
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) if hasattr(vae, 'config') else 8
        
        # 冻结 VAE 和文本编码器
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()
        
    def encode_prompt(self, prompts: List[str], device: torch.device):
        """编码文本提示"""
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]
            
        return prompt_embeds
        
    def encode_images(self, images: torch.Tensor):
        """使用 VAE 编码图像到潜在空间"""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor if hasattr(self.vae, 'config') else latents * 0.18215
        return latents
        
    def decode_latents(self, latents: torch.Tensor):
        """使用 VAE 解码潜在向量到图像"""
        latents = latents / (self.vae.config.scaling_factor if hasattr(self.vae, 'config') else 0.18215)
        
        with torch.no_grad():
            images = self.vae.decode(latents).sample
            
        return images
        
    def forward(self, x, t, tt=None, c=None, prompt_embeds=None):
        """
        前向传播
        
        Args:
            x: 输入数据 (图像或潜在向量) [B, C, H, W] 或 [B, latent_dim]
            t: 当前时间步 [B] 或 [B, 1]
            tt: 目标时间步 [B] 或 [B, 1] (可选)
            c: 条件标签 (可选)
            prompt_embeds: 文本嵌入 [B, seq_len, hidden_dim] (可选)
            
        Returns:
            noise_pred: 预测的噪声或速度场
        """
        device = x.device
        batch_size = x.size(0)
        
        # 处理时间输入
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [B, 1]
            
        # 将时间归一化到 [0, 1]
        timesteps = t.squeeze(1) * 1000  # 假设最大步数为1000
        timesteps = timesteps.long()
        
        # 获取文本嵌入
        if prompt_embeds is None and c is not None:
            # 如果没有提供文本嵌入，使用空文本
            prompt_embeds = torch.zeros(
                batch_size,
                77,  # CLIP 最大序列长度
                768,  # CLIP 隐藏维度
                device=device,
            )
        
        # UNet 前向传播
        noise_pred = self.unet(
            x,
            timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample
        
        return noise_pred


class TwinFlowDiffusersCoreModel(CoreModel):
    """
    基于 Diffusers 的 TwinFlow 核心模型
    """
    
    def __init__(
        self,
        # 模型参数
        unet: Optional[UNet2DConditionModel] = None,
        vae: Optional[AutoencoderKL] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        noise_scheduler: Optional[DDPMScheduler] = None,
        # TwinFlow 参数
        ema_decay_rate: float = 0.99,
        estimate_order: int = 2,
        enhanced_ratio: float = 0.5,
        using_twinflow: bool = True,
        # 训练参数
        learning_rate: float = 2e-4,
        use_vae_encoding: bool = True,
        resolution: int = 512,
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.use_vae_encoding = use_vae_encoding
        self.resolution = resolution
        
        # 创建或保存模型组件
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        
        # 创建 UNet 包装器
        if unet is not None:
            self.model = DiffusersUNetWrapper(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                use_vae_encoding=use_vae_encoding,
                resolution=resolution,
            )
        else:
            self.model = None
            
        # 创建 TwinFlow 训练器
        self.twinflow_trainer = TwinFlowTrainer(
            ema_decay_rate=ema_decay_rate,
            estimate_order=estimate_order,
            enhanced_ratio=enhanced_ratio,
            using_twinflow=using_twinflow,
        )
        
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 解包批次数据
        if isinstance(batch, dict):
            images = batch['pixel_values']
            prompts = batch.get('prompts', [''] * images.size(0))
        else:
            images, prompts = batch
            
        device = next(self.parameters()).device
        images = images.to(device)
        
        # 使用 VAE 编码图像到潜在空间
        if self.use_vae_encoding and self.vae is not None:
            with torch.no_grad():
                latents = self.model.encode_images(images)
        else:
            latents = images
            
        # 编码文本提示
        prompt_embeds = None
        if self.text_encoder is not None:
            prompt_embeds = self.model.encode_prompt(prompts, device)
            
        # 准备条件
        batch_size = latents.size(0)
        c = [torch.zeros(batch_size, dtype=torch.long, device=device)]  # 占位条件
        e = [torch.zeros(batch_size, dtype=torch.long, device=device)]
        
        # 将潜在向量展平以适应 TwinFlow 训练器
        # [B, C, H, W] -> [B, C*H*W]
        if latents.dim() == 4:
            latents_flat = latents.view(latents.size(0), -1)
        else:
            latents_flat = latents
            
        # 使用 TwinFlow 训练器计算损失
        # 注意：这里我们需要包装模型以处理潜在空间
        def latent_model_wrapper(x, t, tt=None, c=None, **kwargs):
            """包装器将展平的潜在向量恢复为图像格式"""
            # x: [B, C*H*W]
            # 恢复为 [B, C, H, W]
            if x.dim() == 2:
                x_reshaped = x.view(latents.shape)
            else:
                x_reshaped = x
                
            # 调用原始模型
            return self.model(x_reshaped, t, tt=tt, c=c, **kwargs)
        
        # 使用 TwinFlow 训练器
        loss = self.twinflow_trainer.training_step(
            model=latent_model_wrapper,
            x=latents_flat,
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
            model=self.unet,
            gradient_clip_val=1.0,
        )
        
        for optimizer in self.optimizers:
            optimizer.step()
        
        for optimizer in self.optimizers:
            optimizer.zero_grad()
            
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 解包批次数据
        if isinstance(batch, dict):
            images = batch['pixel_values']
            prompts = batch.get('prompts', [''] * images.size(0))
        else:
            images, prompts = batch
            
        device = next(self.parameters()).device
        images = images.to(device)
        
        # 只处理少量样本
        images = images[:4]
        prompts = prompts[:4]
        
        with torch.no_grad():
            # 使用 VAE 编码
            if self.use_vae_encoding and self.vae is not None:
                latents = self.model.encode_images(images)
            else:
                latents = images
                
            # 从噪声采样
            z = torch.randn_like(latents)
            
            # 编码文本
            prompt_embeds = None
            if self.text_encoder is not None:
                prompt_embeds = self.model.encode_prompt(prompts, device)
                
            # 使用采样循环生成
            samples = self.twinflow_trainer.sampling_loop(
                inital_noise_z=z,
                sampling_model=self.model,
                sampling_steps=1,  # 单步生成
                c=[torch.zeros(z.size(0), dtype=torch.long, device=device)],
            )
            
            # 解码生成的潜在向量
            generated = self.model.decode_latents(samples[-1])
            
            # 计算重建损失
            recon_loss = F.mse_loss(generated, images)
            
        self.log('val_recon_loss', recon_loss.item())
        
    def configure_optimizers(self):
        """配置优化器"""
        # 只优化 UNet 参数
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        return optimizer
        
    def generate_samples(
        self,
        n_samples: int = 16,
        prompts: Optional[List[str]] = None,
        num_inference_steps: int = 1,
    ):
        """
        生成样本
        
        Args:
            n_samples: 采样数量
            prompts: 文本提示列表，如果为None则使用空字符串
            num_inference_steps: 推理步数（1为单步生成）
            
        Returns:
            images: 生成的图像列表
        """
        device = next(self.parameters()).device
        
        if prompts is None:
            prompts = [''] * n_samples
            
        with torch.no_grad():
            # 编码文本
            prompt_embeds = self.model.encode_prompt(prompts, device)
            
            # 从噪声采样
            z = torch.randn(
                n_samples,
                4,  # latent channels
                self.resolution // 8,
                self.resolution // 8,
                device=device,
            )
            
            # 使用采样循环生成
            samples = self.twinflow_trainer.sampling_loop(
                inital_noise_z=z,
                sampling_model=self.model,
                sampling_steps=num_inference_steps,
                c=[torch.zeros(n_samples, dtype=torch.long, device=device)],
            )
            
            # 解码生成的潜在向量
            images = self.model.decode_latents(samples[-1])
            
        return images


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="TwinFlow Training for Diffusers Models")
    
    # 模型参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="预训练模型路径或HuggingFace模型ID",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="模型修订版本",
    )
    
    # 数据集参数
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="数据集名称或路径",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="训练数据目录",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="图像列名",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="文本列名",
    )
    
    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/twinflow-diffusers",
        help="输出目录",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="图像分辨率",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="最大训练轮数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练",
    )
    
    # TwinFlow 特定参数
    parser.add_argument(
        "--ema_decay_rate",
        type=float,
        default=0.99,
        help="EMA 衰减率",
    )
    parser.add_argument(
        "--estimate_order",
        type=int,
        default=2,
        help="RCGM 估计阶数",
    )
    parser.add_argument(
        "--enhanced_ratio",
        type=float,
        default=0.5,
        help="增强比率 (CFG guidance)",
    )
    parser.add_argument(
        "--no_twinflow",
        action="store_true",
        help="禁用 TwinFlow (仅使用基础 RCGM)",
    )
    
    # 其他参数
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载器工作进程数",
    )
    
    args = parser.parse_args()
    return args


def setup_logging(args):
    """设置日志记录"""
    import logging
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    return logging.getLogger(__name__)


def main():
    """主训练函数"""
    
    # 检查必要的库
    if not DIFFUSERS_AVAILABLE:
        print("错误: 需要安装 diffusers 库")
        print("安装命令: pip install diffusers transformers datasets")
        return
    
    # 解析参数
    args = parse_args()
    logger = setup_logging(args)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查 CUDA 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("TwinFlow Training for Diffusers Models")
    logger.info("=" * 60)
    logger.info(f"预训练模型: {args.pretrained_model_name_or_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"分辨率: {args.resolution}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"最大轮数: {args.max_epochs}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"使用 TwinFlow: {not args.no_twinflow}")
    logger.info("=" * 60)
    
    # TODO: 实现完整的训练流程
    # 包括:
    # 1. 加载预训练模型 (UNet, VAE, Text Encoder)
    # 2. 准备数据集
    # 3. 创建 TwinFlowCoreModel
    # 4. 设置 Trainer
    # 5. 开始训练
    
    logger.info("训练器初始化完成，开始训练...")
    
    # 示例: 这里只是一个框架，实际实现需要完成上述 TODO
    print("\n提示: 这是一个训练框架模板。")
    print("实际使用时需要完成以下内容:")
    print("1. 实现数据加载逻辑")
    print("2. 完成 Diffusers 模型的加载和包装")
    print("3. 实现完整的训练循环")
    print("\n参考实现请查看: train_TwinFlow.py (MNIST 示例)")


if __name__ == "__main__":
    main()
