"""
RMSprop优化器
"""

import torch
from torch.optim import RMSprop as TorchRMSprop


class RMSprop(TorchRMSprop):
    """
    RMSprop优化器 (Root Mean Square Propagation)
    
    Args:
        params: 模型参数或参数字典
        lr: 学习率,默认为0.01
        alpha: 平滑常数,默认为0.99
        eps: 数值稳定性的小常数,默认为1e-8
        weight_decay: 权重衰减(L2正则化),默认为0
        momentum: 动量因子,默认为0
        centered: 是否使用centered RMSprop,默认为False
    """
    
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super().__init__(
            params=params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered
        )


def rmsprop(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
    """
    创建RMSprop优化器的便捷函数
    
    Args:
        params: 模型参数或参数字典
        lr: 学习率,默认为0.01
        alpha: 平滑常数,默认为0.99
        eps: 数值稳定性的小常数,默认为1e-8
        weight_decay: 权重衰减(L2正则化),默认为0
        momentum: 动量因子,默认为0
        centered: 是否使用centered RMSprop,默认为False
    
    Returns:
        RMSprop优化器实例
    """
    return RMSprop(
        params=params,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        centered=centered
    )