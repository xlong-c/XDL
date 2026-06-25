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

# 知识蒸馏损失
from .distillation_loss import (
    DistillationLossBreakdown,
    distillation_loss,
    feature_distillation_loss,
    kl_divergence_with_temperature,
    relation_distillation_loss,
)

# 扩散模型步数蒸馏损失
from .diffusion_distillation_loss import tdm_loss, tdm_loss_weighted

# 偏好优化损失 (DPO / STPO / GRPO)
from .preference_loss import (
    GRPOLossBreakdown,
    STPOLossBreakdown,
    dpo_loss,
    grpo_loss,
    stpo_loss,
)

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

    # 知识蒸馏损失
    register_loss("kl_divergence_with_temperature")(kl_divergence_with_temperature)
    register_loss("distillation_loss")(distillation_loss)
    register_loss("feature_distillation_loss")(feature_distillation_loss)
    register_loss("relation_distillation_loss")(relation_distillation_loss)

    # 扩散模型步数蒸馏损失
    register_loss("tdm_loss")(tdm_loss)
    register_loss("tdm_loss_weighted")(tdm_loss_weighted)

    # 偏好优化损失
    register_loss("dpo_loss")(dpo_loss)
    register_loss("stpo_loss")(stpo_loss)
    register_loss("grpo_loss")(grpo_loss)

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
    "DistillationLossBreakdown",
    "kl_divergence_with_temperature",
    "distillation_loss",
    "feature_distillation_loss",
    "relation_distillation_loss",
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
    "dpo_loss",
    "stpo_loss",
    "grpo_loss",
    "STPOLossBreakdown",
    "GRPOLossBreakdown",
    "tdm_loss",
    "tdm_loss_weighted",
]
