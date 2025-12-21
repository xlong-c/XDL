"""
AdamW优化器
"""

import torch
from torch.optim import AdamW as TorchAdamW


class AdamW(TorchAdamW):
    """
    AdamW优化器 (Adam with decoupled weight decay)
    权重衰减与梯度更新解耦,通常比Adam在正则化方面效果更好
    
    Args:
        params: 模型参数或参数字典
        lr: 学习率,默认为0.001
        betas: 用于计算梯度和梯度平方的移动平均的系数,默认为(0.9, 0.999)
        eps: 数值稳定性的小常数,默认为1e-8
        weight_decay: 权重衰减系数,默认为0.01
        amsgrad: 是否使用AMSGrad变体,默认为False
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )


def adamw(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
    """
    创建AdamW优化器的便捷函数
    
    Args:
        params: 模型参数或参数字典
        lr: 学习率,默认为0.001
        betas: 用于计算梯度和梯度平方的移动平均的系数,默认为(0.9, 0.999)
        eps: 数值稳定性的小常数,默认为1e-8
        weight_decay: 权重衰减系数,默认为0.01
        amsgrad: 是否使用AMSGrad变体,默认为False
    
    Returns:
        AdamW优化器实例
    """
    return AdamW(
        params=params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad
    )