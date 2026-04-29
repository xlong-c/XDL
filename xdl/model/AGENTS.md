# XDL Models — 模型架构模块

**Updated:** 2026-04-29
**Branch:** master

## 概述

XDL 模型模块包含常用深度学习网络架构，通过注册系统支持配置化构建。当前共 **30 个**注册条目。

## 架构

```
xdl/model/
├── __init__.py         # 统一注册入口 (30 条目)
├── resnet.py           # ResNet 系列 (18/34/50/101/152)
├── vgg.py              # VGG 系列 (11/13/16/19, BN 变体)
├── vit.py              # Vision Transformer 系列 (Tiny→Huge)
├── simple_mlp.py       # 简单 MLP
├── generate/           # 生成模型
│   └── twinflow.py     # TwinFlow 连续时间生成模型
└── segment/            # 分割模型
    └── fatt.py         # FATT 分割模型
```

## 已注册模型 (30 条目)

### ResNet 系列 (8)

| 模型 | 注册名 | 说明 |
|------|--------|------|
| BasicBlock | `BasicBlock` | 基础残差块 (18/34 层) |
| Bottleneck | `Bottleneck` | 瓶颈残差块 (50/101/152 层) |
| ResNet (基类) | `ResNet` | 可自定义配置 |
| ResNet-18 | `resnet18` | 轻量级 |
| ResNet-34 | `resnet34` | 中轻量 |
| ResNet-50 | `resnet50` | 标准 |
| ResNet-101 | `resnet101` | 深层 |
| ResNet-152 | `resnet152` | 极深 |

### VGG 系列 (9)

| 模型 | 注册名 | 说明 |
|------|--------|------|
| VGG (基类) | `VGG` | 可自定义配置 |
| VGG-11 | `vgg11` | 无 BN |
| VGG-11-BN | `vgg11_bn` | 带 BN |
| VGG-13 | `vgg13` | 无 BN |
| VGG-13-BN | `vgg13_bn` | 带 BN |
| VGG-16 | `vgg16` | 无 BN |
| VGG-16-BN | `vgg16_bn` | 带 BN |
| VGG-19 | `vgg19` | 无 BN |
| VGG-19-BN | `vgg19_bn` | 带 BN |

### Vision Transformer 系列 (9)

| 模型 | 注册名 | 参数量 |
|------|--------|--------|
| PatchEmbedding | `PatchEmbedding` | 图像分块嵌入 |
| MultiHeadAttention | `MultiHeadAttention` | 多头注意力 |
| TransformerBlock | `TransformerBlock` | Transformer 块 |
| VisionTransformer (基类) | `VisionTransformer` | 可自定义配置 |
| ViT-Tiny | `vit_tiny_patch16_224` | ~5M |
| ViT-Small | `vit_small_patch16_224` | ~22M |
| ViT-Base | `vit_base_patch16_224` | ~86M |
| ViT-Large | `vit_large_patch16_224` | ~307M |
| ViT-Huge | `vit_huge_patch14_224` | ~632M |

### MLP (2)

| 模型 | 注册名 | 说明 |
|------|--------|------|
| SimpleMLP (类) | `SimpleMLP` | 可自定义隐藏层维度 |
| simple_mlp (工厂) | `simple_mlp` | 默认配置 (784→256→128→10) |

### 生成模型 (1)

| 模型 | 注册名 | 说明 |
|------|--------|------|
| TwinFlow | `TwinFlow` | 连续时间生成模型 (RCGM + 一致性正则化) |

### 分割模型 (1)

| 模型 | 注册名 | 说明 |
|------|--------|------|
| FATT | `FATT` | Feature-Aware Transformer 分割模型 |

## 使用方式

### 纯代码方式

```python
from xdl.model import resnet50, vit_base_patch16_224, SimpleMLP, TwinFlow, FATT

# CNN
model = resnet50(num_classes=100)
# ViT
vit = vit_base_patch16_224(num_classes=1000)
# MLP
mlp = SimpleMLP(in_features=784, hidden_dims=[512, 256], out_features=10)
```

### YAML 配置方式（推荐）

```python
from xdl.utils.registry import build_model

model = build_model("resnet50", num_classes=100)
```

```yaml
# ResNet
model:
  target: "registry:resnet50"
  params:
    num_classes: 100
```

```yaml
# TwinFlow — 连续时间生成模型
model:
  target: "registry:TwinFlow"
  params:
    consistc_ratio: 1.0       # 一致性正则化权重
    ema_decay_rate: 0.99      # EMA 衰减率
    estimate_order: 2         # ODE 求解阶数 (1 或 2)
    num_sampling_steps: 20    # 采样步数
    sde_correction: true      # SDE 随机校正
    dist_match_weight: 0.1    # 分布匹配权重
    lr: 0.001                 # 学习率
    beta1: 0.9                # Adam beta1
```

## 开发规范

### 注册新模型

```python
from xdl.utils.registry import register_model
import torch.nn as nn

@register_model("MyModel")
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 实现
```

### 模型命名约定

- 类名：`PascalCase`（如 `ResNet`, `VisionTransformer`, `FATT`）
- 工厂函数：`snake_case`（如 `resnet50`, `vit_base_patch16_224`）
- 注册名：与类名或工厂函数一致

### 必须实现

1. `__init__`：接受 `num_classes` 参数
2. `forward`：标准前向传播
3. 在 `xdl/model/__init__.py` 中 `import`、`register`、`__all__` 导出

## 注意事项

- **输入尺寸**：ViT 系列默认 224×224（Huge 为 14 patch, 224 输入）
- **预训练权重**：当前未提供，需自行加载
- **segment/ 目录**：FATT 已注册可用，非预留状态
- **TwinFlow**：详见模块 docstring，支持 ODE/SDE 采样、RCGM 整流、分布匹配
