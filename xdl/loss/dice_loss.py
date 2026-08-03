"""Dice 损失 — 图像分割常用损失。"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice 损失，用于分割任务。

    Args:
        smooth: 平滑项，防止除零，默认 1.0。
        average: 'macro' 对每类平均, 'micro' 全局计算，默认 'macro'。
    """

    def __init__(self, smooth: float = 1.0, average: str = "macro"):
        super().__init__()
        self.smooth = smooth
        self.average = average

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Dice 损失。

        Args:
            pred: (B, C, ...) 概率或 logit，会自动 softmax（二分类用 sigmoid）。
            target: (B, ...) 整数标签 或 (B, C, ...) one-hot。
        """
        num_classes = pred.shape[1]

        if num_classes == 1:
            pred_prob = torch.sigmoid(pred)
            target_onehot = target.float() if target.dim() == pred_prob.dim() else target.unsqueeze(1).float()
        else:
            pred_prob = torch.softmax(pred, dim=1)
            if target.dim() < pred_prob.dim():
                target_onehot = F_onehot(target, num_classes).to(pred.dtype)
            else:
                target_onehot = target.float()

        intersection = (pred_prob * target_onehot).sum(dim=(0, 2, 3))
        cardinality = pred_prob.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.average == "micro":
            intersection_all = intersection.sum()
            cardinality_all = cardinality.sum()
            dice = (2.0 * intersection_all + self.smooth) / (cardinality_all + self.smooth)
        else:
            dice = dice_per_class.mean()

        return 1.0 - dice


class GeneralizedDiceLoss(nn.Module):
    """广义 Dice 损失，按类频率加权以处理不平衡分割。

    Args:
        smooth: 平滑项，默认 1.0。
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]

        if num_classes == 1:
            pred_prob = torch.sigmoid(pred)
            target_onehot = target.float() if target.dim() == pred_prob.dim() else target.unsqueeze(1).float()
        else:
            pred_prob = torch.softmax(pred, dim=1)
            if target.dim() < pred_prob.dim():
                target_onehot = F_onehot(target, num_classes).to(pred.dtype)
            else:
                target_onehot = target.float()

        intersection = (pred_prob * target_onehot).sum(dim=(0, 2, 3))
        cardinality = pred_prob.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

        # 类别权重 = 1 / (target 中各类像素数的平方)
        w = 1.0 / (target_onehot.sum(dim=(0, 2, 3)) ** 2 + self.smooth)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - (w * dice_per_class).sum() / w.sum()


def F_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """将标签转为 one-hot 编码。"""
    return torch.nn.functional.one_hot(labels.long(), num_classes).permute(0, 3, 1, 2).float()
