"""
损失函数模块 - 包含常见的深度学习损失函数
"""

from ..utils.registry import register_loss

# Focal Loss 损失函数
from .focal_loss import BinaryFocalLoss, FocalLoss, binary_focal_loss, focal_loss

# Huber Loss
from .huber_loss import HuberLoss

# 对比学习损失
from .contrastive_loss import InfoNCE, NTXentLoss

# 分割损失
from .dice_loss import DiceLoss, GeneralizedDiceLoss


def _register_losses():
    """统一注册所有损失函数到 LOSS_REGISTRY"""

    # Focal Loss 系列
    register_loss("FocalLoss")(FocalLoss)
    register_loss("BinaryFocalLoss")(BinaryFocalLoss)
    register_loss("focal_loss")(focal_loss)
    register_loss("binary_focal_loss")(binary_focal_loss)

    # 回归损失
    register_loss("HuberLoss")(HuberLoss)

    # 对比学习损失
    register_loss("InfoNCE")(InfoNCE)
    register_loss("NTXentLoss")(NTXentLoss)

    # 分割损失
    register_loss("DiceLoss")(DiceLoss)
    register_loss("GeneralizedDiceLoss")(GeneralizedDiceLoss)


# 自动执行损失函数注册
_register_losses()

__all__ = [
    "FocalLoss",
    "BinaryFocalLoss",
    "focal_loss",
    "binary_focal_loss",
    "HuberLoss",
    "InfoNCE",
    "NTXentLoss",
    "DiceLoss",
    "GeneralizedDiceLoss",
]
