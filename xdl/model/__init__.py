"""
模型模块 - 包含常见的深度学习网络架构
"""

import importlib
import logging

from ..utils.registry import register_model

logger = logging.getLogger(__name__)

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
from .simple_mlp import SimpleMLP, simple_mlp

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
from .generate import (
    DistributionMatcher,
    IdentityRepresentor,
    RepresentationScattering,
    RepresentationScatteringField,
    ScatteringTracker,
    TBSMGenerator,
    TwinFlow,
)
from .segment.fatt import FATT
from .lowlevel import (
    RGT,
    ATD,
    RRDBNet,
    OFTSR_UNet,
    OFTSR_SuperResModel,
    AutoEncoder_RRDBNet,
    ProbabilisticAutoEncoder_RRDBNet,
    realplksr,
)


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
    register_model("resnet18")(resnet18)
    register_model("resnet34")(resnet34)
    register_model("resnet50")(resnet50)
    register_model("resnet101")(resnet101)
    register_model("resnet152")(resnet152)

    # 注册Vision Transformer系列模型
    register_model("VisionTransformer")(VisionTransformer)
    register_model("vit_tiny_patch16_224")(vit_tiny_patch16_224)
    register_model("vit_small_patch16_224")(vit_small_patch16_224)
    register_model("vit_base_patch16_224")(vit_base_patch16_224)
    register_model("vit_large_patch16_224")(vit_large_patch16_224)
    register_model("vit_huge_patch14_224")(vit_huge_patch14_224)

    # 注册生成模型
    register_model("TwinFlow")(TwinFlow)
    register_model("TBSMGenerator")(TBSMGenerator)
    register_model("IdentityRepresentor")(IdentityRepresentor)
    register_model("ScatteringTracker")(ScatteringTracker)
    register_model("RepresentationScatteringField")(RepresentationScatteringField)
    register_model("DistributionMatcher")(DistributionMatcher)
    register_model("RepresentationScattering")(RepresentationScattering)

    # 注册简单 MLP
    register_model("SimpleMLP")(SimpleMLP)
    register_model("simple_mlp")(simple_mlp)
    register_model("FATT")(FATT)

    # 注册 RGT 超分辨率模型
    register_model("RGT")(RGT)
    register_model("RGT_S")(RGT)  # RGT small 变体，共用 RGT 类

    # 注册 ATD 超分辨率模型
    register_model("ATD")(ATD)

    # 注册 RRDBNet (ESRGAN 生成器)
    register_model("RRDBNet")(RRDBNet)

    # 注册 OFTSR Flow-based SR 模型
    register_model("OFTSR_UNet")(OFTSR_UNet)
    register_model("OFTSR_SuperResModel")(OFTSR_SuperResModel)

    # 注册 AESOP AutoEncoder 模型
    register_model("AutoEncoder_RRDBNet")(AutoEncoder_RRDBNet)
    register_model("ProbabilisticAutoEncoder_RRDBNet")(ProbabilisticAutoEncoder_RRDBNet)

    # 注册 RealPLKSR 超分辨率模型 (仅依赖核心依赖)
    register_model("RealPLKSR")(realplksr)


# 研究型模型: 依赖较重或对 torch 版本敏感, 导入失败时显式告警并跳过.
_OPTIONAL_MODELS = (
    ("RealPLKSR_Ult", "xdl.model.lowlevel.realplksr_arch_ult", "realplksr"),
    ("WFEN", "xdl.model.lowlevel.wfen_arch", "WFEN"),
    ("DAT_2", "xdl.model.lowlevel.dat_arch", "dat_2"),
    ("ESC", "xdl.model.lowlevel.esc_arch", "ESC"),
)


def _register_optional_models() -> None:
    """注册依赖较重或版本敏感的研究模型; 依赖缺失时显式告警并跳过."""
    for name, module_path, attribute in _OPTIONAL_MODELS:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning("跳过注册 %s (%s): %s", name, module_path, exc)
            continue
        register_model(name)(getattr(module, attribute))


# 自动执行模型注册
_register_models()
_register_optional_models()

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
    "SimpleMLP",
    "simple_mlp",
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
    "TBSMGenerator",
    "IdentityRepresentor",
    "ScatteringTracker",
    "RepresentationScatteringField",
    "DistributionMatcher",
    "RepresentationScattering",
    # Segment
    "FATT",
    # Lowlevel SR
    "RGT",
    "ATD",
    "RRDBNet",
    "OFTSR_UNet",
    "OFTSR_SuperResModel",
    "AutoEncoder_RRDBNet",
    "ProbabilisticAutoEncoder_RRDBNet",
]
