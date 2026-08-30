"""
损失函数模块 - 包含常见的深度学习损失函数
"""

from ..utils.registry import register_loss

# 分类损失
from .classification_loss import (
    AsymmetricLoss,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)

# 检测框损失
from .box_loss import BoxIoULoss

# Focal Loss 损失函数
from .focal_loss import BinaryFocalLoss, FocalLoss, binary_focal_loss, focal_loss

# Huber Loss
from .huber_loss import HuberLoss

# 图像重建损失
from .reconstruction_loss import (
    CharbonnierLoss,
    GradientDifferenceLoss,
    ReconstructionLoss,
    SSIMLoss,
    TotalVariationLoss,
)

# 图像生成损失
from .generative_loss import (
    DiffusionPredictionLoss,
    FeatureMatchingLoss,
    GANLoss,
    HingeDiscriminatorLoss,
    HingeGeneratorLoss,
    KLDivergenceLoss,
    VAELoss,
)

# 对比学习损失
from .contrastive_loss import InfoNCE, NTXentLoss

# 知识蒸馏, 步数蒸馏与偏好优化损失 (DPO / STPO / GRPO) 已收拢到
# xdl.post_training 子模块, 注册见 xdl/post_training/__init__.py

# 分割损失
from .dice_loss import DiceLoss, GeneralizedDiceLoss
from .segmentation_loss import (
    DiceCrossEntropyLoss,
    FocalTverskyLoss,
    JaccardLoss,
    TverskyLoss,
)

# NLP 序列损失
from .sequence_loss import (
    CausalLanguageModelingLoss,
    MaskedCrossEntropyLoss,
    SequenceCrossEntropyLoss,
    TokenClassificationLoss,
)


def _register_losses():
    """统一注册所有损失函数到 LOSS_REGISTRY"""

    # 分类损失
    register_loss("LabelSmoothingCrossEntropy")(LabelSmoothingCrossEntropy)
    register_loss("SoftTargetCrossEntropy")(SoftTargetCrossEntropy)
    register_loss("AsymmetricLoss")(AsymmetricLoss)

    # Focal Loss 系列
    register_loss("FocalLoss")(FocalLoss)
    register_loss("BinaryFocalLoss")(BinaryFocalLoss)
    register_loss("focal_loss")(focal_loss)
    register_loss("binary_focal_loss")(binary_focal_loss)

    # 回归损失
    register_loss("HuberLoss")(HuberLoss)

    # 图像重建损失
    register_loss("CharbonnierLoss")(CharbonnierLoss)
    register_loss("TotalVariationLoss")(TotalVariationLoss)
    register_loss("GradientDifferenceLoss")(GradientDifferenceLoss)
    register_loss("SSIMLoss")(SSIMLoss)
    register_loss("ReconstructionLoss")(ReconstructionLoss)

    # 图像生成损失
    register_loss("KLDivergenceLoss")(KLDivergenceLoss)
    register_loss("VAELoss")(VAELoss)
    register_loss("GANLoss")(GANLoss)
    register_loss("HingeDiscriminatorLoss")(HingeDiscriminatorLoss)
    register_loss("HingeGeneratorLoss")(HingeGeneratorLoss)
    register_loss("FeatureMatchingLoss")(FeatureMatchingLoss)
    register_loss("DiffusionPredictionLoss")(DiffusionPredictionLoss)

    # 对比学习损失
    register_loss("InfoNCE")(InfoNCE)
    register_loss("NTXentLoss")(NTXentLoss)

    # 知识蒸馏 / 步数蒸馏 / 偏好优化损失注册已迁移到 xdl.post_training

    # 分割损失
    register_loss("DiceLoss")(DiceLoss)
    register_loss("GeneralizedDiceLoss")(GeneralizedDiceLoss)
    register_loss("JaccardLoss")(JaccardLoss)
    register_loss("TverskyLoss")(TverskyLoss)
    register_loss("FocalTverskyLoss")(FocalTverskyLoss)
    register_loss("DiceCrossEntropyLoss")(DiceCrossEntropyLoss)

    # 检测框回归损失
    register_loss("BoxIoULoss")(BoxIoULoss)

    # NLP 序列损失
    register_loss("MaskedCrossEntropyLoss")(MaskedCrossEntropyLoss)
    register_loss("SequenceCrossEntropyLoss")(SequenceCrossEntropyLoss)
    register_loss("TokenClassificationLoss")(TokenClassificationLoss)
    register_loss("CausalLanguageModelingLoss")(CausalLanguageModelingLoss)


# 自动执行损失函数注册
_register_losses()

__all__ = [
    "LabelSmoothingCrossEntropy",
    "SoftTargetCrossEntropy",
    "AsymmetricLoss",
    "FocalLoss",
    "BinaryFocalLoss",
    "focal_loss",
    "binary_focal_loss",
    "HuberLoss",
    "CharbonnierLoss",
    "TotalVariationLoss",
    "GradientDifferenceLoss",
    "SSIMLoss",
    "ReconstructionLoss",
    "KLDivergenceLoss",
    "VAELoss",
    "GANLoss",
    "HingeDiscriminatorLoss",
    "HingeGeneratorLoss",
    "FeatureMatchingLoss",
    "DiffusionPredictionLoss",
    "InfoNCE",
    "NTXentLoss",
    "DiceLoss",
    "GeneralizedDiceLoss",
    "JaccardLoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    "DiceCrossEntropyLoss",
    "BoxIoULoss",
    "MaskedCrossEntropyLoss",
    "SequenceCrossEntropyLoss",
    "TokenClassificationLoss",
    "CausalLanguageModelingLoss",
]
