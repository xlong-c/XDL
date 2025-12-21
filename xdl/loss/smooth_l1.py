"""
Smooth L1损失函数
"""

import torch
import torch.nn as nn


class SmoothL1Loss(nn.SmoothL1Loss):
    """
    Smooth L1损失函数,也称为Huber Loss,用于回归任务
    在目标检测中广泛使用,对异常值更鲁棒
    
    Args:
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
        beta: 平滑参数,默认为1.0
    """
    
    def __init__(self, size_average=None, reduce=None, reduction='mean', beta=1.0):
        super().__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            beta=beta
        )


class HuberLoss(SmoothL1Loss):
    """
    Huber Loss的别名,与SmoothL1Loss相同
    """
    pass


def smooth_l1_loss(size_average=None, reduce=None, reduction='mean', beta=1.0):
    """
    创建Smooth L1损失函数的便捷函数
    
    Args:
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
        beta: 平滑参数,默认为1.0
    
    Returns:
        SmoothL1Loss损失函数实例
    """
    return SmoothL1Loss(
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        beta=beta
    )


def huber_loss(size_average=None, reduce=None, reduction='mean', beta=1.0):
    """
    创建Huber损失函数的便捷函数
    
    Args:
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
        beta: 平滑参数,默认为1.0
    
    Returns:
        HuberLoss损失函数实例
    """
    return HuberLoss(
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        beta=beta
    )