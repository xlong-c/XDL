"""
SGD优化器
"""

import torch
from torch.optim import SGD as TorchSGD


class SGD(TorchSGD):
    """
    随机梯度下降优化器
    
    Args:
        params: 模型参数或参数字典
        lr: 学习率,默认为0.001
        momentum: 动量因子,默认为0
        dampening: 动量阻尼,默认为0
        weight_decay: 权重衰减(L2正则化),默认为0
        nesterov: 是否启用Nesterov动量,默认为False
    """
    
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )


def sgd(params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    """
    创建SGD优化器的便捷函数
    
    Args:
        params: 模型参数或参数字典
        lr: 学习率,默认为0.001
        momentum: 动量因子,默认为0
        dampening: 动量阻尼,默认为0
        weight_decay: 权重衰减(L2正则化),默认为0
        nesterov: 是否启用Nesterov动量,默认为False
    
    Returns:
        SGD优化器实例
    """
    return SGD(
        params=params,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov
    )