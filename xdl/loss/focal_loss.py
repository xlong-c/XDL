"""
Focal Loss损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss损失函数,用于解决类别不平衡问题
    主要用于目标检测任务,如RetinaNet
    
    Args:
        alpha: 类别权重,float或tensor,默认为1.0
        gamma: 聚焦参数,减少简单样本的权重,默认为2.0
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
        ignore_index: 忽略的类别索引,默认为-100
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        """
        前向传播
        
        Args:
            inputs: 预测logits,shape为(N, C)
            targets: 真实标签,shape为(N,)
        
        Returns:
            计算的focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    二元Focal Loss损失函数,用于二分类的类别不平衡问题
    
    Args:
        alpha: 正样本权重,默认为1.0
        gamma: 聚焦参数,默认为2.0
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        前向传播
        
        Args:
            inputs: 预测logits,shape为(N,)
            targets: 真实标签,shape为(N,)
        
        Returns:
            计算的binary focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def focal_loss(alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
    """
    创建Focal Loss损失函数的便捷函数
    
    Args:
        alpha: 类别权重,float或tensor,默认为1.0
        gamma: 聚焦参数,默认为2.0
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
        ignore_index: 忽略的类别索引,默认为-100
    
    Returns:
        FocalLoss损失函数实例
    """
    return FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index
    )


def binary_focal_loss(alpha=1.0, gamma=2.0, reduction='mean'):
    """
    创建二元Focal Loss损失函数的便捷函数
    
    Args:
        alpha: 正样本权重,默认为1.0
        gamma: 聚焦参数,默认为2.0
        reduction: 损失的约减方式,'none' | 'mean' | 'sum',默认为'mean'
    
    Returns:
        BinaryFocalLoss损失函数实例
    """
    return BinaryFocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction
    )