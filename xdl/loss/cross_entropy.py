"""
交叉熵损失函数
"""

import torch.nn as nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    交叉熵损失函数,用于多分类任务
    
    Args:
        weight: 类别权重,shape为(num_classes,)的tensor,默认为None
        size_average: 已弃用,使用reduction参数
        ignore_index: 忽略的类别索引,默认为-100
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
        label_smoothing: 标签平滑系数,范围[0.0, 1.0],默认为0.0
    """
    
    def __init__(self, weight=None, size_average=None, ignore_index=-100, 
                 reduce=None, reduction='mean', label_smoothing=0.0):
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing
        )


def cross_entropy_loss(weight=None, size_average=None, ignore_index=-100, 
                      reduce=None, reduction='mean', label_smoothing=0.0):
    """
    创建交叉熵损失函数的便捷函数
    
    Args:
        weight: 类别权重,shape为(num_classes,)的tensor,默认为None
        size_average: 已弃用,使用reduction参数
        ignore_index: 忽略的类别索引,默认为-100
        reduce: 已弃用,使用reduction参数
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
        label_smoothing: 标签平滑系数,范围[0.0, 1.0],默认为0.0
    
    Returns:
        CrossEntropyLoss损失函数实例
    """
    return CrossEntropyLoss(
        weight=weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing
    )