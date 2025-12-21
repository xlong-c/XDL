"""
损失函数模块 - 包含常见的深度学习损失函数
"""

from ..utils.registry import register_loss

# 交叉熵损失函数
from .cross_entropy import CrossEntropyLoss, cross_entropy_loss

# 均方误差损失函数
from .mse import MSELoss, mse_loss

# 平均绝对误差损失函数
from .mae import L1Loss, MAELoss, l1_loss, mae_loss

# 二元交叉熵损失函数
from .bce import BCELoss, BCEWithLogitsLoss, bce_loss, bce_with_logits_loss

# Smooth L1损失函数
from .smooth_l1 import SmoothL1Loss, HuberLoss, smooth_l1_loss, huber_loss

# Focal Loss损失函数
from .focal_loss import FocalLoss, BinaryFocalLoss, focal_loss, binary_focal_loss


def _register_losses():
    """统一注册所有损失函数到LOSS_REGISTRY"""
    
    # 注册交叉熵损失函数系列
    register_loss("CrossEntropyLoss")(CrossEntropyLoss)
    register_loss("cross_entropy_loss")(cross_entropy_loss)
    
    # 注册均方误差损失函数系列
    register_loss("MSELoss")(MSELoss)
    register_loss("mse_loss")(mse_loss)
    
    # 注册平均绝对误差损失函数系列
    register_loss("L1Loss")(L1Loss)
    register_loss("MAELoss")(MAELoss)
    register_loss("l1_loss")(l1_loss)
    register_loss("mae_loss")(mae_loss)
    
    # 注册二元交叉熵损失函数系列
    register_loss("BCELoss")(BCELoss)
    register_loss("BCEWithLogitsLoss")(BCEWithLogitsLoss)
    register_loss("bce_loss")(bce_loss)
    register_loss("bce_with_logits_loss")(bce_with_logits_loss)
    
    # 注册Smooth L1损失函数系列
    register_loss("SmoothL1Loss")(SmoothL1Loss)
    register_loss("HuberLoss")(HuberLoss)
    register_loss("smooth_l1_loss")(smooth_l1_loss)
    register_loss("huber_loss")(huber_loss)
    
    # 注册Focal Loss损失函数系列
    register_loss("FocalLoss")(FocalLoss)
    register_loss("BinaryFocalLoss")(BinaryFocalLoss)
    register_loss("focal_loss")(focal_loss)
    register_loss("binary_focal_loss")(binary_focal_loss)


# 自动执行损失函数注册
_register_losses()

__all__ = [
    # 交叉熵损失函数
    'CrossEntropyLoss', 'cross_entropy_loss',
    
    # 均方误差损失函数
    'MSELoss', 'mse_loss',
    
    # 平均绝对误差损失函数
    'L1Loss', 'MAELoss', 'l1_loss', 'mae_loss',
    
    # 二元交叉熵损失函数
    'BCELoss', 'BCEWithLogitsLoss', 'bce_loss', 'bce_with_logits_loss',
    
    # Smooth L1损失函数
    'SmoothL1Loss', 'HuberLoss', 'smooth_l1_loss', 'huber_loss',
    
    # Focal Loss损失函数
    'FocalLoss', 'BinaryFocalLoss', 'focal_loss', 'binary_focal_loss'
]