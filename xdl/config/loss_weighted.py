"""
加权损失函数
支持多个损失函数的加权组合
"""

from typing import List, Optional

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
    
    def __init__(self, losses: List[nn.Module], weights: Optional[List[float]] = None):
        """
        初始化加权损失函数
        
        Args:
            losses: 损失函数列表
            weights: 权重列表，如果为 None 则均匀加权
        """
        super().__init__()
        if not losses:
            raise ValueError("losses cannot be empty")

        self.losses = nn.ModuleList(losses)
        if weights is None:
            normalized_weights = [1.0 / len(losses)] * len(losses)
        else:
            if len(weights) != len(losses):
                raise ValueError("weights length must match losses length")
            normalized_weights = list(weights)
        
        # 归一化权重
        total = sum(normalized_weights)
        if total == 0:
            raise ValueError("weights sum cannot be zero")

        self.weights = [weight / total for weight in normalized_weights]

    @staticmethod
    def _compute_loss(
        loss_fn: nn.Module,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        调用单个损失函数并校验返回类型。
        """
        loss_value = loss_fn(pred, target)
        if not isinstance(loss_value, torch.Tensor):
            raise TypeError("loss function must return torch.Tensor")
        return loss_value
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算加权损失
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            加权损失值
        """
        first_loss = self._compute_loss(self.losses[0], pred, target)
        total_loss = self.weights[0] * first_loss
        for loss_fn, weight in zip(self.losses[1:], self.weights[1:]):
            loss_value = self._compute_loss(loss_fn, pred, target)
            total_loss = total_loss + weight * loss_value
        return total_loss
    
    def __repr__(self) -> str:
        loss_names = [type(loss_module).__name__ for loss_module in self.losses]
        return f"WeightedLoss({loss_names}, weights={self.weights})"
