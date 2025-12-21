"""
平均绝对误差损失函数
"""

import torch
import torch.nn as nn


class L1Loss(nn.L1Loss):
    """
    平均绝对误差损失函数(L1Loss),也称为MAE,用于回归任务
    
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


class MAELoss(L1Loss):
    """
    平均绝对误差损失函数的别名
    """
    pass


def l1_loss(size_average=None, reduce=None, reduction='mean'):
    """
    创建L1损失函数的便捷函数
    
    Args:
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
    
    Returns:
        L1Loss损失函数实例
    """
    return L1Loss(
        size_average=size_average,
        reduce=reduce,
        reduction=reduction
    )


def mae_loss(size_average=None, reduce=None, reduction='mean'):
    """
    创建MAE损失函数的便捷函数
    
    Args:
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
    
    Returns:
        MAELoss损失函数实例
    """
    return MAELoss(
        size_average=size_average,
        reduce=reduce,
        reduction=reduction
    )