"""
损失函数模块 - 包含常见的深度学习损失函数
"""

from ..utils.registry import register_loss

# Focal Loss损失函数
from .focal_loss import FocalLoss, BinaryFocalLoss, focal_loss, binary_focal_loss


def _register_losses():
    """统一注册所有损失函数到LOSS_REGISTRY"""
    
    # 注册Focal Loss损失函数系列
    register_loss("FocalLoss")(FocalLoss)
    register_loss("BinaryFocalLoss")(BinaryFocalLoss)
    register_loss("focal_loss")(focal_loss)
    register_loss("binary_focal_loss")(binary_focal_loss)


# 自动执行损失函数注册
_register_losses()

__all__ = [
    # Focal Loss损失函数
    'FocalLoss', 'BinaryFocalLoss', 'focal_loss', 'binary_focal_loss'
]
