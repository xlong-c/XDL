"""
加权损失函数
支持多个损失函数的加权组合
"""

from typing import List

import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    """
    加权损失函数
    
    将多个损失函数进行加权组合，适用于多任务学习等场景。
    
    Example:
        >>> loss_fn = WeightedLoss([
        ...     nn.MSELoss(),
        ...     nn.L1Loss()
        ... ], weights=[0.7, 0.3])
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, losses: List[nn.Module], weights: List[float] = None):
        """
        初始化加权损失函数
        
        Args:
            losses: 损失函数列表
            weights: 权重列表，如果为 None 则均匀加权
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0 / len(losses)] * len(losses)
        
        # 归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算加权损失
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            加权损失值
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss
    
    def __repr__(self) -> str:
        loss_names = [type(l).__name__ for l in self.losses]
        return f"WeightedLoss({loss_names}, weights={self.weights})"
