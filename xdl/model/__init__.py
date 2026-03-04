"""
模型模块 - 包含常见的深度学习网络架构
"""

from ..utils.registry import register_model

# ResNet网络
from .resnet import (
    BasicBlock,
    Bottleneck,
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

# VGG网络
from .vgg import VGG, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn

# Vision Transformer网络
from .vit import (
    MultiHeadAttention,
    PatchEmbedding,
    TransformerBlock,
    VisionTransformer,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch16_224,
    vit_tiny_patch16_224,
)
from .generate import TwinFlow


def _register_models():
    """统一注册所有模型到MODEL_REGISTRY"""

    # 注册VGG系列模型
    register_model("VGG")(VGG)
    register_model("vgg11")(vgg11)
    register_model("vgg11_bn")(vgg11_bn)
    register_model("vgg13")(vgg13)
    register_model("vgg13_bn")(vgg13_bn)
    register_model("vgg16")(vgg16)
    register_model("vgg16_bn")(vgg16_bn)
    register_model("vgg19")(vgg19)
    register_model("vgg19_bn")(vgg19_bn)

    # 注册ResNet系列模型
    register_model("ResNet")(ResNet)
    register_model("BasicBlock")(BasicBlock)
    register_model("Bottleneck")(Bottleneck)
    register_model("resnet18")(resnet18)
    register_model("resnet34")(resnet34)
    register_model("resnet50")(resnet50)
    register_model("resnet101")(resnet101)
    register_model("resnet152")(resnet152)

    # 注册Vision Transformer系列模型
    register_model("VisionTransformer")(VisionTransformer)
    register_model("PatchEmbedding")(PatchEmbedding)
    register_model("MultiHeadAttention")(MultiHeadAttention)
    register_model("TransformerBlock")(TransformerBlock)
    register_model("vit_tiny_patch16_224")(vit_tiny_patch16_224)
    register_model("vit_small_patch16_224")(vit_small_patch16_224)
    register_model("vit_base_patch16_224")(vit_base_patch16_224)
    register_model("vit_large_patch16_224")(vit_large_patch16_224)
    register_model("vit_huge_patch14_224")(vit_huge_patch14_224)

    # 注册生成模型
    register_model("TwinFlow")(TwinFlow)


# 自动执行模型注册
_register_models()

__all__ = [
    # VGG
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    # ResNet
    "ResNet",
    "BasicBlock",
    "Bottleneck",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    # Vision Transformer
    "VisionTransformer",
    "PatchEmbedding",
    "MultiHeadAttention",
    "TransformerBlock",
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "vit_huge_patch14_224",
    # Generate
    "TwinFlow",
]
