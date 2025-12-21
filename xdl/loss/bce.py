"""
二元交叉熵损失函数
"""

import torch
import torch.nn as nn


class BCELoss(nn.BCELoss):
    """
    二元交叉熵损失函数,用于二分类任务
    输入需要先通过sigmoid函数
    
    Args:
        weight: 样本权重,shape为(N,)的tensor,默认为None
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
    """
    
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    带logits的二元交叉熵损失函数,用于二分类任务
    内部包含sigmoid函数,数值稳定性更好
    
    Args:
        weight: 样本权重,shape为(N,)的tensor,默认为None
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
        pos_weight: 正样本权重,shape为(C,)的tensor,默认为None
    """
    
    def __init__(self, weight=None, size_average=None, reduce=None, 
                 reduction='mean', pos_weight=None):
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight
        )


def bce_loss(weight=None, size_average=None, reduce=None, reduction='mean'):
    """
    创建二元交叉熵损失函数的便捷函数
    
    Args:
        weight: 样本权重,shape为(N,)的tensor,默认为None
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
    
    Returns:
        BCELoss损失函数实例
    """
    return BCELoss(
        weight=weight,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction
    )


def bce_with_logits_loss(weight=None, size_average=None, reduce=None, 
                        reduction='mean', pos_weight=None):
    """
    创建带logits的二元交叉熵损失函数的便捷函数
    
    Args:
        weight: 样本权重,shape为(N,)的tensor,默认为None
        size_average: 已弃用,使用reduction参数
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
        pos_weight: 正样本权重,shape为(C,)的tensor,默认为None
    
    Returns:
        BCEWithLogitsLoss损失函数实例
    """
    return BCEWithLogitsLoss(
        weight=weight,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        pos_weight=pos_weight
    )