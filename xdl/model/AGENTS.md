# XDL Models - 模型架构模块

**Generated:** 2026-02-27
**Commit:** d3b4406
**Branch:** master

## 概述

XDL 模型模块包含常用深度学习网络架构，通过注册系统支持配置化构建。

## 架构

```
xdl/model/
├── __init__.py      # 统一注册入口
├── resnet.py        # ResNet 系列 (18/34/50/101/152)
├── vgg.py           # VGG 系列 (11/13/16/19, BN 变体)
├── vit.py           # Vision Transformer 系列
├── generate/        # 生成模型
│   └── twinflow.py  # TwinFlow (512行)
└── segment/         # 分割模型（预留）
```

## 已注册模型

### CNN 系列

| 模型 | 注册名 | 说明 |
|------|--------|------|
| ResNet-18 | `resnet18` | 轻量级 |
| ResNet-50 | `resnet50` | 标准 |
| ResNet-152 | `resnet152` | 深层 |
| VGG-16 | `vgg16` | 无 BN |
| VGG-16-BN | `vgg16_bn` | 带 BN |

### Vision Transformer 系列

| 模型 | 注册名 | 参数量 |
|------|--------|--------|
| ViT-Tiny | `vit_tiny_patch16_224` | ~5M |
| ViT-Small | `vit_small_patch16_224` | ~22M |
| ViT-Base | `vit_base_patch16_224` | ~86M |
| ViT-Large | `vit_large_patch16_224` | ~307M |

### 生成模型

| 模型 | 注册名 | 说明 |
|------|--------|------|
| TwinFlow | `TwinFlow` | 双流生成模型 |

## 使用方式

### 纯代码方式

```python
from xdl.model import resnet50, VisionTransformer

model = resnet50(num_classes=100)
vit = vit_base_patch16_224(num_classes=1000)
```

### 配置文件方式

```python
from xdl.utils.registry import build_model

model = build_model("resnet50", num_classes=100)
```

```yaml
# config.yaml
model:
  name: "resnet50"
  params:
    num_classes: 100
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

- 类名：`PascalCase`（如 `ResNet`, `VisionTransformer`）
- 工厂函数：`snake_case`（如 `resnet50`, `vit_base_patch16_224`）
- 注册名：与工厂函数一致

### 必须实现

1. `__init__`：接受 `num_classes` 参数
2. `forward`：标准前向传播
3. 在 `__init__.py` 中注册和导出

## 模型组件

### ViT 核心组件

- `PatchEmbedding`：图像分块嵌入
- `MultiHeadAttention`：多头注意力
- `TransformerBlock`：Transformer 块

### ResNet 核心组件

- `BasicBlock`：基础残差块（18/34层）
- `Bottleneck`：瓶颈块（50/101/152层）

## 注意事项

- **输入尺寸**：ViT 系列默认 224x224
- **预训练权重**：当前未提供，需自行加载
- **模型导出**：新模型必须在 `__init__.py` 的 `__all__` 中声明