"""
Vision Transformer (ViT) network implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """图像patch嵌入层"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层进行patch嵌入
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入图像大小 ({H}x{W}) 不匹配模型配置 ({self.img_size}x{self.img_size})"
        
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class MLP(nn.Module):
    """前馈网络"""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, dropout: float = 0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0., attn_dropout: float = 0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力 + 残差连接
        x = x + self.attn(self.norm1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer主网络"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, 
                 attn_dropout: float = 0.):
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # 类别token和位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 初始化位置嵌入
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化其他参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取"""
        B = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        
        # 添加类别token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, n_patches+1, embed_dim]
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer编码器
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x[:, 0]  # 返回类别token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# 预定义的ViT模型配置
def vit_tiny_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Tiny/16模型"""
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        num_classes=num_classes, **kwargs
    )
    return model


def vit_small_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Small/16模型"""
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        num_classes=num_classes, **kwargs
    )
    return model


def vit_base_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Base/16模型"""
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        num_classes=num_classes, **kwargs
    )
    return model


def vit_large_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Large/16模型"""
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        num_classes=num_classes, **kwargs
    )
    return model


def vit_huge_patch14_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Huge/14模型"""
    model = VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        num_classes=num_classes, **kwargs
    )
    return model