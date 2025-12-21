"""
均方误差损失函数
"""

import torch
import torch.nn as nn


class MSELoss(nn.MSELoss):
    """
    均方误差损失函数,用于回归任务
    
    Args:
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
    """
    
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )


def mse_loss(size_average=None, reduce=None, reduction='mean'):
    """
    创建均方误差损失函数的便捷函数
    
    Args:
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
    
    Returns:
        MSELoss损失函数实例
    """
    return MSELoss(
        size_average=size_average,
        reduce=reduce,
        reduction=reduction
    )