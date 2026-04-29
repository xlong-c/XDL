"""Huber 损失 — 对离群值稳健的回归损失。"""

import torch.nn as nn


class HuberLoss(nn.Module):
    """Huber 损失，在 delta 范围内为 MSE，超出为 MAE。

    Args:
        delta: 阈值，默认 1.0。
        reduction: 'none' | 'mean' | 'sum'，默认 'mean'。
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self._loss = nn.HuberLoss(reduction=reduction, delta=delta)

    def forward(self, inputs, targets):
        return self._loss(inputs, targets)
