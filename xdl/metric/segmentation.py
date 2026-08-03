"""Segmentation metrics."""

from __future__ import annotations

import torch

from xdl.metric._utils import (
    _segmentation_labels
)


class IoU:
    """
    IoU(交并比)指标 — 支持二分类和多分类分割任务。

    __call__ 自动检测输入形状:
    - 二分类: pred/target 同为 2D/3D (B, H, W) 或含单通道 (B, 1, H, W)
    - 多分类: pred (B, C, H, W) + target (B, H, W), C>1 时自动触发 per-class 平均
    """

    def __init__(
        self,
        num_classes: int | None = None,
        average: str = "macro",
        threshold: float = 0.5,
        epsilon: float = 1e-6,
    ):
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.epsilon = epsilon

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        # 多分类: pred 4D 且通道数 > 1 → per-class macro IoU
        if pred.dim() == 4 and pred.shape[1] > 1:
            num_classes = self.num_classes or pred.shape[1]
            if target.dim() != 3:
                raise ValueError(f"多分类 target 应为 (B, H, W), 实际 {target.shape}")
            pred_labels = torch.argmax(pred, dim=1)
            iou_sum = torch.tensor(0.0, device=target.device)
            for class_idx in range(num_classes):
                pred_mask = (pred_labels == class_idx).float()
                target_mask = (target == class_idx).float()
                intersection = (pred_mask * target_mask).sum()
                union = pred_mask.sum() + target_mask.sum() - intersection
                if union > 0:
                    iou_sum += intersection / (union + self.epsilon)
            return (iou_sum / num_classes).item()

        # 二分类
        if pred.shape != target.shape:
            raise ValueError(
                f"预测和真实标签形状不匹配: {pred.shape} vs {target.shape}"
            )
        if pred.dim() > 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() > 2 and target.shape[1] == 1:
            target = target.squeeze(1)
        pred_binary = (pred > self.threshold).float()
        target_binary = target.float()
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        return (intersection / (union + self.epsilon)).item()

    def __call_multi_class__(
        self, pred: torch.Tensor, target: torch.Tensor, num_classes: int
    ) -> float:
        """Deprecated: 请直接使用 __call__, 自动检测多分类。"""
        return self(pred, target)


class DiceCoefficient:
    """
    Dice系数指标 — 支持二分类和多分类分割任务。

    __call__ 自动检测输入形状:
    - 二分类: pred/target 同为 2D/3D (B, H, W) 或含单通道 (B, 1, H, W)
    - 多分类: pred (B, C, H, W) + target (B, H, W), C>1 时自动触发 per-class 平均
    """

    def __init__(
        self,
        num_classes: int | None = None,
        average: str = "macro",
        threshold: float = 0.5,
        epsilon: float = 1e-6,
    ):
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.epsilon = epsilon

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        # 多分类: pred 4D 且通道数 > 1 → per-class macro Dice
        if pred.dim() == 4 and pred.shape[1] > 1:
            num_classes = self.num_classes or pred.shape[1]
            if target.dim() != 3:
                raise ValueError(f"多分类 target 应为 (B, H, W), 实际 {target.shape}")
            pred_labels = torch.argmax(pred, dim=1)
            dice_sum = torch.tensor(0.0, device=target.device)
            for class_idx in range(num_classes):
                pred_mask = (pred_labels == class_idx).float()
                target_mask = (target == class_idx).float()
                intersection = (pred_mask * target_mask).sum()
                total = pred_mask.sum() + target_mask.sum()
                if total > 0:
                    dice_sum += (2 * intersection) / (total + self.epsilon)
            return (dice_sum / num_classes).item()

        # 二分类
        if pred.shape != target.shape:
            raise ValueError(
                f"预测和真实标签形状不匹配: {pred.shape} vs {target.shape}"
            )
        if pred.dim() > 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() > 2 and target.shape[1] == 1:
            target = target.squeeze(1)
        pred_binary = (pred > self.threshold).float()
        target_binary = target.float()
        intersection = (pred_binary * target_binary).sum()
        total = pred_binary.sum() + target_binary.sum()
        return ((2 * intersection) / (total + self.epsilon)).item()

    def __call_multi_class__(
        self, pred: torch.Tensor, target: torch.Tensor, num_classes: int
    ) -> float:
        """Deprecated: 请直接使用 __call__, 自动检测多分类。"""
        return self(pred, target)


class PixelAccuracy:
    """语义分割 pixel accuracy, 支持 ignore_index."""

    def __init__(self, ignore_index: int | None = None, threshold: float = 0.5) -> None:
        self.ignore_index = ignore_index
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if not torch.is_floating_point(pred):
            pred_labels = pred.long()
        elif pred.dim() >= 4 and pred.shape[1] > 1:
            pred_labels = torch.argmax(pred, dim=1)
        else:
            if pred.dim() == target.dim() + 1 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            pred_labels = (pred > self.threshold).long()
        target_labels = target.long()
        valid = torch.ones_like(target_labels, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = target_labels != self.ignore_index
        if not torch.any(valid):
            return 0.0
        return (pred_labels[valid] == target_labels[valid]).float().mean().item()


class MeanIoU:
    """Mean IoU for semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        threshold: float = 0.5,
        epsilon: float = 1e-6,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self.union = torch.zeros(self.num_classes, dtype=torch.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_labels = (
            _segmentation_labels(pred, target, self.threshold).reshape(-1).cpu()
        )
        target_labels = target.long().reshape(-1).cpu()
        valid = (target_labels >= 0) & (target_labels < self.num_classes)
        if self.ignore_index is not None:
            valid = valid & (target_labels != self.ignore_index)
        pred_labels = pred_labels[valid]
        target_labels = target_labels[valid]
        for class_idx in range(self.num_classes):
            pred_mask = pred_labels == class_idx
            target_mask = target_labels == class_idx
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            self.intersection[class_idx] += intersection
            self.union[class_idx] += union

    def compute(self) -> float:
        valid = self.union > 0
        if not torch.any(valid):
            return 0.0
        iou = self.intersection[valid] / (self.union[valid] + self.epsilon)
        return iou.mean().item()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        self.reset()
        self.update(pred, target)
        return self.compute()


class FrequencyWeightedIoU(MeanIoU):
    """Frequency weighted IoU for semantic segmentation."""

    def reset(self) -> None:
        super().reset()
        self.target_count = torch.zeros(self.num_classes, dtype=torch.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_labels = (
            _segmentation_labels(pred, target, self.threshold).reshape(-1).cpu()
        )
        target_labels = target.long().reshape(-1).cpu()
        valid = (target_labels >= 0) & (target_labels < self.num_classes)
        if self.ignore_index is not None:
            valid = valid & (target_labels != self.ignore_index)
        pred_labels = pred_labels[valid]
        target_labels = target_labels[valid]
        for class_idx in range(self.num_classes):
            pred_mask = pred_labels == class_idx
            target_mask = target_labels == class_idx
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            self.intersection[class_idx] += intersection
            self.union[class_idx] += union
            self.target_count[class_idx] += target_mask.sum().item()

    def compute(self) -> float:
        total = self.target_count.sum()
        if total == 0:
            return 0.0
        iou = self.intersection / (self.union + self.epsilon)
        weights = self.target_count / total
        return (weights * iou).sum().item()

