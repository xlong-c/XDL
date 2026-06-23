"""深度学习评估指标模块."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


class Accuracy:
    """
    准确率指标
    支持多分类和二分类
    """

    def __init__(self, num_classes: int | None = None, threshold: float = 0.5):
        """
        Args:
            num_classes: 分类数量,如果为None则自动判断为二分类或多分类
            threshold: 二分类时的阈值
        """
        self.num_classes = num_classes
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算准确率

        Args:
            pred: 预测值,形状为 (batch_size, num_classes) 或 (batch_size,)
            target: 真实标签,形状为 (batch_size,)

        Returns:
            准确率 (0-1之间的浮点数)
        """
        if pred.dim() > 1 and pred.shape[1] > 1:
            # 多分类情况
            pred_labels = torch.argmax(pred, dim=1)
        else:
            # 二分类情况
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            pred_labels = (pred >= self.threshold).long()

        correct = (pred_labels == target).sum().float()
        total = torch.tensor(target.shape[0], dtype=torch.float, device=target.device)
        return (correct / total).item()


class Precision:
    """
    精确率指标 — 支持 macro/micro 平均。
    """

    def __init__(self, num_classes: int | None = None, average: str = "macro"):
        self.num_classes = num_classes
        self.average = average

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.dim() > 1 and pred.shape[1] > 1:
            pred_labels = torch.argmax(pred, dim=1)
        else:
            pred_labels = (pred >= 0.5).long()

        if self.num_classes is None:
            self.num_classes = len(torch.unique(target))

        # micro: 全局 TP / (TP + FP)
        if self.average == "micro":
            tp = torch.tensor(0.0, device=target.device)
            fp = torch.tensor(0.0, device=target.device)
            for class_idx in range(self.num_classes):
                pred_pos = pred_labels == class_idx
                true_pos = target == class_idx
                tp += (pred_pos & true_pos).sum().float()
                fp += (pred_pos & ~true_pos).sum().float()
            return (tp / (tp + fp)).item() if tp + fp > 0 else 0.0

        # macro: 逐类平均
        precision_sum = torch.tensor(0.0, device=target.device)
        for class_idx in range(self.num_classes):
            pred_pos = pred_labels == class_idx
            true_pos = target == class_idx
            tp = (pred_pos & true_pos).sum().float()
            fp = (pred_pos & ~true_pos).sum().float()
            if tp + fp > 0:
                precision_sum += tp / (tp + fp)
        return (precision_sum / self.num_classes).item()


class Recall:
    """
    召回率指标 — 支持 macro/micro 平均。
    """

    def __init__(self, num_classes: int | None = None, average: str = "macro"):
        self.num_classes = num_classes
        self.average = average

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.dim() > 1 and pred.shape[1] > 1:
            pred_labels = torch.argmax(pred, dim=1)
        else:
            pred_labels = (pred >= 0.5).long()

        if self.num_classes is None:
            self.num_classes = len(torch.unique(target))

        # micro: 全局 TP / (TP + FN)
        if self.average == "micro":
            tp = torch.tensor(0.0, device=target.device)
            fn = torch.tensor(0.0, device=target.device)
            for class_idx in range(self.num_classes):
                pred_pos = pred_labels == class_idx
                true_pos = target == class_idx
                tp += (pred_pos & true_pos).sum().float()
                fn += (~pred_pos & true_pos).sum().float()
            return (tp / (tp + fn)).item() if tp + fn > 0 else 0.0

        # macro: 逐类平均
        recall_sum = torch.tensor(0.0, device=target.device)
        for class_idx in range(self.num_classes):
            pred_pos = pred_labels == class_idx
            true_pos = target == class_idx
            tp = (pred_pos & true_pos).sum().float()
            fn = (~pred_pos & true_pos).sum().float()
            if tp + fn > 0:
                recall_sum += tp / (tp + fn)
        return (recall_sum / self.num_classes).item()


class F1Score:
    """
    F1分数指标 — 支持 macro/micro 平均。
    """

    def __init__(self, num_classes: int | None = None, average: str = "macro"):
        self.num_classes = num_classes
        self.average = average

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.dim() > 1 and pred.shape[1] > 1:
            pred_labels = torch.argmax(pred, dim=1)
        else:
            pred_labels = (pred >= 0.5).long()

        if self.num_classes is None:
            self.num_classes = len(torch.unique(target))

        # micro: 全局聚合后算 F1
        if self.average == "micro":
            tp = torch.tensor(0.0, device=target.device)
            fp = torch.tensor(0.0, device=target.device)
            fn = torch.tensor(0.0, device=target.device)
            for class_idx in range(self.num_classes):
                pred_pos = pred_labels == class_idx
                true_pos = target == class_idx
                tp += (pred_pos & true_pos).sum().float()
                fp += (pred_pos & ~true_pos).sum().float()
                fn += (~pred_pos & true_pos).sum().float()
            if tp + fp + fn == 0:
                return 0.0
            return (tp / (tp + 0.5 * (fp + fn))).item()

        # macro: 逐类 F1 平均
        f1_sum = torch.tensor(0.0, device=target.device)
        for class_idx in range(self.num_classes):
            pred_pos = pred_labels == class_idx
            true_pos = target == class_idx
            tp = (pred_pos & true_pos).sum().float()
            fp = (pred_pos & ~true_pos).sum().float()
            fn = (~pred_pos & true_pos).sum().float()
            precision = (
                tp / (tp + fp)
                if tp + fp > 0
                else torch.tensor(0.0, device=target.device)
            )
            recall = (
                tp / (tp + fn)
                if tp + fn > 0
                else torch.tensor(0.0, device=target.device)
            )
            if precision + recall > 0:
                f1_sum += 2 * (precision * recall) / (precision + recall)
        return (f1_sum / self.num_classes).item()


class MeanAbsoluteError:
    """
    平均绝对误差指标
    """

    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算平均绝对误差

        Args:
            pred: 预测值
            target: 真实值

        Returns:
            平均绝对误差
        """
        return torch.mean(torch.abs(pred - target)).item()


class MeanSquaredError:
    """
    均方误差指标
    """

    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算均方误差

        Args:
            pred: 预测值
            target: 真实值

        Returns:
            均方误差
        """
        return torch.mean((pred - target) ** 2).item()


class RootMeanSquaredError:
    """
    均方根误差指标
    """

    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算均方根误差

        Args:
            pred: 预测值
            target: 真实值

        Returns:
            均方根误差
        """
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()


class TopKAccuracy:
    """
    Top-K准确率指标
    """

    def __init__(self, k: int = 5):
        """
        Args:
            k: Top-K中的K值
        """
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算Top-K准确率

        Args:
            pred: 预测值,形状为 (batch_size, num_classes)
            target: 真实标签,形状为 (batch_size,)

        Returns:
            Top-K准确率
        """
        if pred.dim() < 2:
            pred_labels = (pred.reshape(-1) > 0).long()
            return (pred_labels == target.reshape(-1).long()).float().mean().item()
        _, pred_topk = torch.topk(pred, min(self.k, pred.shape[-1]), dim=-1)
        target_expanded = target.reshape(-1, 1).long().expand_as(pred_topk)
        correct = (pred_topk == target_expanded).any(dim=1).sum().float()
        total = torch.tensor(target.shape[0], dtype=torch.float, device=target.device)
        return (correct / total).item()


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


def _to_label_predictions(
    pred: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    if not torch.is_floating_point(pred):
        return pred.long()
    if pred.dim() > 1 and pred.shape[1] > 1:
        return torch.argmax(pred, dim=1)
    if pred.dim() > 1:
        pred = pred.squeeze(-1)
    return (pred >= threshold).long()


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp(min=torch.finfo(torch.float32).eps),
        torch.zeros_like(numerator),
    )


def _confusion_matrix_from_labels(
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if pred_labels.numel() != target_labels.numel():
        raise ValueError("pred and target must contain the same number of labels")
    valid = (target_labels >= 0) & (target_labels < num_classes)
    valid = valid & (pred_labels >= 0) & (pred_labels < num_classes)
    encoded = target_labels[valid] * num_classes + pred_labels[valid]
    counts = torch.bincount(encoded.cpu(), minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)


class ConfusionMatrix:
    """累积分类混淆矩阵.

    Args:
        num_classes: 类别数量.
        threshold: 二分类 logits/probability 的阈值.
    """

    def __init__(self, num_classes: int, threshold: float = 0.5) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_labels = _to_label_predictions(pred.detach(), self.threshold).reshape(-1)
        target_labels = target.detach().long().reshape(-1)
        self.matrix += _confusion_matrix_from_labels(
            pred_labels.cpu(),
            target_labels.cpu(),
            self.num_classes,
        )

    def compute(self) -> torch.Tensor:
        return self.matrix.clone()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.reset()
        self.update(pred, target)
        return self.compute()


class BalancedAccuracy:
    """Balanced accuracy, 即各类召回率的平均值."""

    def __init__(self, num_classes: int | None = None, threshold: float = 0.5) -> None:
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._confusion: torch.Tensor | None = None

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_labels = _to_label_predictions(pred.detach(), self.threshold).reshape(-1)
        target_labels = target.detach().long().reshape(-1)
        if pred_labels.numel() != target_labels.numel():
            raise ValueError("pred and target must contain the same number of labels")
        num_classes = self.num_classes
        if num_classes is None:
            max_label = torch.cat([pred_labels, target_labels]).max().item()
            num_classes = int(max_label) + 1
        batch_matrix = _confusion_matrix_from_labels(
            pred_labels.cpu(),
            target_labels.cpu(),
            num_classes,
        )
        if self._confusion is None:
            self._confusion = batch_matrix
        else:
            if self._confusion.shape != batch_matrix.shape:
                raise ValueError("num_classes changed across updates")
            self._confusion += batch_matrix

    def compute(self) -> float:
        if self._confusion is None:
            return 0.0
        confusion = self._confusion.to(dtype=torch.float32)
        recalls = _safe_divide(confusion.diag(), confusion.sum(dim=1))
        valid = confusion.sum(dim=1) > 0
        if not torch.any(valid):
            return 0.0
        return recalls[valid].mean().item()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        self.reset()
        self.update(pred, target)
        return self.compute()


class MatthewsCorrCoef:
    """Matthews correlation coefficient for binary or multiclass classification."""

    def __init__(self, num_classes: int | None = None, threshold: float = 0.5) -> None:
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._confusion: torch.Tensor | None = None

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_labels = _to_label_predictions(pred.detach(), self.threshold).reshape(-1)
        target_labels = target.detach().long().reshape(-1)
        if pred_labels.numel() != target_labels.numel():
            raise ValueError("pred and target must contain the same number of labels")
        num_classes = self.num_classes
        if num_classes is None:
            max_label = torch.cat([pred_labels, target_labels]).max().item()
            num_classes = int(max_label) + 1
        batch_matrix = _confusion_matrix_from_labels(
            pred_labels.cpu(),
            target_labels.cpu(),
            num_classes,
        )
        if self._confusion is None:
            self._confusion = batch_matrix
        else:
            if self._confusion.shape != batch_matrix.shape:
                raise ValueError("num_classes changed across updates")
            self._confusion += batch_matrix

    def compute(self) -> float:
        if self._confusion is None:
            return 0.0
        confusion = self._confusion.to(dtype=torch.float64)
        total = confusion.sum()
        if total == 0:
            return 0.0
        true_positive = confusion.diag()
        predicted = confusion.sum(dim=0)
        actual = confusion.sum(dim=1)
        numerator = true_positive.sum() * total - (predicted * actual).sum()
        denominator = torch.sqrt(
            (total**2 - (predicted**2).sum()) * (total**2 - (actual**2).sum())
        )
        if denominator == 0:
            return 0.0
        return (numerator / denominator).item()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        self.reset()
        self.update(pred, target)
        return self.compute()


class MultiLabelAccuracy:
    """多标签逐元素准确率."""

    def __init__(self, threshold: float = 0.5, from_logits: bool = True) -> None:
        self.threshold = threshold
        self.from_logits = from_logits

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
            )
        prob = torch.sigmoid(pred) if self.from_logits else pred
        pred_binary = prob >= self.threshold
        target_binary = target.bool()
        return (pred_binary == target_binary).float().mean().item()


class MultiLabelPrecision:
    """多标签 precision, 支持 micro/macro/samples 平均."""

    def __init__(
        self,
        threshold: float = 0.5,
        average: str = "micro",
        from_logits: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        if average not in {"micro", "macro", "samples"}:
            raise ValueError("average must be one of: 'micro', 'macro', 'samples'")
        self.threshold = threshold
        self.average = average
        self.from_logits = from_logits
        self.epsilon = epsilon

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        stats = _multilabel_stats(pred, target, self.threshold, self.from_logits)
        return _multilabel_score(stats, self.average, "precision", self.epsilon)


class MultiLabelRecall:
    """多标签 recall, 支持 micro/macro/samples 平均."""

    def __init__(
        self,
        threshold: float = 0.5,
        average: str = "micro",
        from_logits: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        if average not in {"micro", "macro", "samples"}:
            raise ValueError("average must be one of: 'micro', 'macro', 'samples'")
        self.threshold = threshold
        self.average = average
        self.from_logits = from_logits
        self.epsilon = epsilon

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        stats = _multilabel_stats(pred, target, self.threshold, self.from_logits)
        return _multilabel_score(stats, self.average, "recall", self.epsilon)


class MultiLabelF1Score:
    """多标签 F1, 支持 micro/macro/samples 平均."""

    def __init__(
        self,
        threshold: float = 0.5,
        average: str = "micro",
        from_logits: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        if average not in {"micro", "macro", "samples"}:
            raise ValueError("average must be one of: 'micro', 'macro', 'samples'")
        self.threshold = threshold
        self.average = average
        self.from_logits = from_logits
        self.epsilon = epsilon

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        stats = _multilabel_stats(pred, target, self.threshold, self.from_logits)
        return _multilabel_score(stats, self.average, "f1", self.epsilon)


def _multilabel_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    from_logits: bool,
) -> dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
        )
    if pred.dim() < 2:
        raise ValueError("multi-label tensors must have shape (N, C, ...)")
    prob = torch.sigmoid(pred) if from_logits else pred
    pred_binary = prob >= threshold
    target_binary = target.bool()
    dims = tuple(dim for dim in range(pred.dim()) if dim != 1)
    sample_dims = tuple(range(1, pred.dim()))
    return {
        "tp_class": (pred_binary & target_binary).sum(dim=dims).float(),
        "fp_class": (pred_binary & ~target_binary).sum(dim=dims).float(),
        "fn_class": (~pred_binary & target_binary).sum(dim=dims).float(),
        "tp_sample": (pred_binary & target_binary).sum(dim=sample_dims).float(),
        "fp_sample": (pred_binary & ~target_binary).sum(dim=sample_dims).float(),
        "fn_sample": (~pred_binary & target_binary).sum(dim=sample_dims).float(),
    }


def _multilabel_score(
    stats: dict[str, torch.Tensor],
    average: str,
    score: str,
    epsilon: float,
) -> float:
    if average == "micro":
        tp = stats["tp_class"].sum()
        fp = stats["fp_class"].sum()
        fn = stats["fn_class"].sum()
        return _precision_recall_f1(tp, fp, fn, score, epsilon).item()
    if average == "macro":
        values = _precision_recall_f1(
            stats["tp_class"], stats["fp_class"], stats["fn_class"], score, epsilon
        )
        return values.mean().item()
    values = _precision_recall_f1(
        stats["tp_sample"], stats["fp_sample"], stats["fn_sample"], score, epsilon
    )
    return values.mean().item()


def _precision_recall_f1(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
    score: str,
    epsilon: float,
) -> torch.Tensor:
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    if score == "precision":
        return precision
    if score == "recall":
        return recall
    return 2.0 * precision * recall / (precision + recall + epsilon)


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


def _segmentation_labels(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if not torch.is_floating_point(pred):
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
            )
        return pred.long()
    if pred.dim() == target.dim() + 1 and pred.shape[1] > 1:
        return torch.argmax(pred, dim=1)
    if pred.dim() == target.dim() + 1 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
        )
    return (pred > threshold).long()


class BoxIoU:
    """Aligned bounding box IoU metric."""

    def __init__(
        self,
        box_format: str = "xyxy",
        reduction: str = "mean",
        epsilon: float = 1e-7,
    ) -> None:
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
        self.box_format = box_format
        self.reduction = reduction
        self.epsilon = epsilon

    def __call__(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> float | torch.Tensor:
        iou = _aligned_box_iou(pred_boxes, target_boxes, self.box_format, self.epsilon)
        if self.reduction == "none":
            return iou
        if self.reduction == "sum":
            return iou.sum().item()
        return iou.mean().item()


def _prediction_tensor(item: Any, key: str) -> torch.Tensor:
    if isinstance(item, Mapping):
        value = item[key]
    else:
        value = getattr(item, key)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


class DetectionMeanAveragePrecision:
    """COCO-style mean average precision over decoded detection predictions.

    ``predictions`` must provide per-image ``boxes``, ``scores`` and ``labels``
    fields, either as mappings or objects with attributes. ``targets`` must
    provide ``boxes`` and ``labels``.
    """

    def __init__(
        self,
        iou_thresholds: Sequence[float] | None = None,
        box_format: str = "xyxy",
        epsilon: float = 1e-7,
    ) -> None:
        self.iou_thresholds = list(
            iou_thresholds
            if iou_thresholds is not None
            else [0.5 + 0.05 * index for index in range(10)]
        )
        self.box_format = box_format
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.predictions: list[Any] = []
        self.targets: list[Mapping[str, Any]] = []

    def update(
        self,
        predictions: Sequence[Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> None:
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def compute(self) -> dict[str, float]:
        aps = [
            self._average_precision(self.predictions, self.targets, float(threshold))
            for threshold in self.iou_thresholds
        ]
        metric_map = {
            "map50_95": float(sum(aps) / len(aps)) if aps else 0.0,
            "map50": 0.0,
            "map75": 0.0,
        }
        for threshold, ap in zip(self.iou_thresholds, aps, strict=False):
            if abs(float(threshold) - 0.5) < 1e-6:
                metric_map["map50"] = float(ap)
            if abs(float(threshold) - 0.75) < 1e-6:
                metric_map["map75"] = float(ap)
        return metric_map

    def __call__(
        self,
        predictions: Sequence[Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, float]:
        self.reset()
        self.update(predictions, targets)
        return self.compute()

    def _build_tp_flags(
        self,
        prediction: Any,
        target: Mapping[str, Any],
        iou_threshold: float,
    ) -> tuple[torch.Tensor, int]:
        pred_boxes = _prediction_tensor(prediction, "boxes").float()
        pred_scores = _prediction_tensor(prediction, "scores").float()
        pred_labels = _prediction_tensor(prediction, "labels").long()
        target_boxes = _prediction_tensor(target, "boxes").float()
        target_labels = _prediction_tensor(target, "labels").long()
        if pred_boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool), int(target_boxes.shape[0])

        matched_targets = torch.zeros((target_boxes.shape[0],), dtype=torch.bool)
        tp = torch.zeros((pred_boxes.shape[0],), dtype=torch.bool)
        order = pred_scores.argsort(descending=True)
        sorted_boxes = pred_boxes[order]
        sorted_labels = pred_labels[order]
        for pred_position, (box, label) in enumerate(
            zip(sorted_boxes, sorted_labels, strict=False)
        ):
            label_mask = target_labels == label
            if not bool(label_mask.any()):
                continue
            candidate_indices = torch.nonzero(label_mask, as_tuple=False).reshape(-1)
            ious = _aligned_box_iou(
                box.unsqueeze(0).expand(candidate_indices.shape[0], -1),
                target_boxes[candidate_indices],
                self.box_format,
                self.epsilon,
            )
            best_iou, best_index = ious.max(dim=0)
            candidate_index = int(candidate_indices[int(best_index.item())].item())
            if float(best_iou.item()) >= iou_threshold and not bool(
                matched_targets[candidate_index]
            ):
                tp[pred_position] = True
                matched_targets[candidate_index] = True
        reordered = torch.zeros_like(tp)
        reordered[order] = tp
        return reordered, int(target_boxes.shape[0])

    def _average_precision(
        self,
        predictions: Sequence[Any],
        targets: Sequence[Mapping[str, Any]],
        iou_threshold: float,
    ) -> float:
        all_scores: list[torch.Tensor] = []
        all_tp: list[torch.Tensor] = []
        total_gt = 0
        for prediction, target in zip(predictions, targets, strict=False):
            tp_flags, gt_count = self._build_tp_flags(
                prediction,
                target,
                iou_threshold,
            )
            all_scores.append(_prediction_tensor(prediction, "scores").detach().cpu())
            all_tp.append(tp_flags.detach().cpu())
            total_gt += gt_count
        if total_gt == 0:
            return 0.0
        scores = torch.cat(all_scores) if all_scores else torch.empty((0,))
        tp = torch.cat(all_tp) if all_tp else torch.empty((0,), dtype=torch.bool)
        if scores.numel() == 0:
            return 0.0
        order = scores.argsort(descending=True)
        tp_sorted = tp[order].float()
        fp_sorted = 1.0 - tp_sorted
        tp_cum = tp_sorted.cumsum(dim=0)
        fp_cum = fp_sorted.cumsum(dim=0)
        recall = tp_cum / max(total_gt, 1)
        precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0)
        precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        for index in range(precision.numel() - 1, 0, -1):
            precision[index - 1] = torch.maximum(precision[index - 1], precision[index])
        delta = recall[1:] - recall[:-1]
        return float((delta * precision[1:]).sum().item())


def _boxes_to_xyxy(boxes: torch.Tensor, box_format: str) -> torch.Tensor:
    if boxes.shape[-1] != 4:
        raise ValueError("boxes must end with 4 coordinates")
    if box_format == "xyxy":
        return boxes
    if box_format != "cxcywh":
        raise ValueError("box_format must be 'xyxy' or 'cxcywh'")
    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ),
        dim=-1,
    )


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    size = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0.0)
    return size[..., 0] * size[..., 1]


def _aligned_box_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    box_format: str,
    epsilon: float,
) -> torch.Tensor:
    if pred_boxes.shape != target_boxes.shape:
        raise ValueError(
            f"pred_boxes and target_boxes shape mismatch: {pred_boxes.shape} vs {target_boxes.shape}"
        )
    pred_xyxy = _boxes_to_xyxy(pred_boxes, box_format)
    target_xyxy = _boxes_to_xyxy(target_boxes, box_format)
    inter_top_left = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    inter_bottom_right = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    inter_size = (inter_bottom_right - inter_top_left).clamp(min=0.0)
    intersection = inter_size[..., 0] * inter_size[..., 1]
    union = _box_area(pred_xyxy) + _box_area(target_xyxy) - intersection
    return intersection / union.clamp(min=epsilon)


class Perplexity:
    """Perplexity for language modeling logits."""

    def __init__(
        self, ignore_index: int = -100, max_value: float | None = None
    ) -> None:
        self.ignore_index = ignore_index
        self.max_value = max_value
        self.reset()

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.token_count = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.dim() != target.dim() + 1:
            raise ValueError("logits must have one more dimension than target")
        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab_size)
        flat_target = target.reshape(-1).long()
        valid = flat_target != self.ignore_index
        if not torch.any(valid):
            return
        losses = torch.nn.functional.cross_entropy(
            flat_logits[valid],
            flat_target[valid],
            reduction="sum",
        )
        self.loss_sum += float(losses.item())
        self.token_count += int(valid.sum().item())

    def compute(self) -> float:
        if self.token_count == 0:
            return 0.0
        value = math.exp(self.loss_sum / self.token_count)
        if self.max_value is not None:
            return min(value, self.max_value)
        return value

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        self.reset()
        self.update(logits, target)
        return self.compute()


class TokenAccuracy:
    """Token-level accuracy for sequence labeling or language modeling."""

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.dim() != target.dim() + 1:
            raise ValueError("logits must have one more dimension than target")
        pred = torch.argmax(logits, dim=-1)
        valid = target != self.ignore_index
        if not torch.any(valid):
            return
        self.correct += int((pred[valid] == target[valid]).sum().item())
        self.total += int(valid.sum().item())

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        self.reset()
        self.update(logits, target)
        return self.compute()


class SequenceExactMatch:
    """Exact-match ratio for generated or labeled sequences."""

    def __init__(self, ignore_index: int | None = None) -> None:
        self.ignore_index = ignore_index

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.dim() == target.dim() + 1:
            pred_labels = torch.argmax(pred, dim=-1)
        else:
            pred_labels = pred.long()
        if pred_labels.shape != target.shape:
            raise ValueError(
                f"pred and target shape mismatch: {pred_labels.shape} vs {target.shape}"
            )
        if self.ignore_index is None:
            matches = (pred_labels == target).all(dim=-1)
            return matches.float().mean().item()
        valid = target != self.ignore_index
        token_matches = (pred_labels == target) | ~valid
        sequence_has_token = valid.any(dim=-1)
        exact = token_matches.all(dim=-1) & sequence_has_token
        if not torch.any(sequence_has_token):
            return 0.0
        return exact.float().sum().item() / sequence_has_token.float().sum().item()


class MeanReciprocalRank:
    """Mean reciprocal rank for retrieval or ranking logits."""

    def __call__(self, scores: torch.Tensor, target: torch.Tensor) -> float:
        if scores.dim() != 2 or target.dim() != 1:
            raise ValueError("scores must be (N, C) and target must be (N,)")
        order = torch.argsort(scores, dim=1, descending=True)
        target_expanded = target.long().unsqueeze(1)
        matches = order == target_expanded
        if not torch.all(matches.any(dim=1)):
            raise ValueError("every target must be a valid class index in scores")
        ranks = matches.float().argmax(dim=1).float() + 1.0
        return (1.0 / ranks).mean().item()


class HitRateAtK:
    """Hit rate@K for retrieval or ranking logits."""

    def __init__(self, k: int = 10) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def __call__(self, scores: torch.Tensor, target: torch.Tensor) -> float:
        if scores.dim() != 2 or target.dim() != 1:
            raise ValueError("scores must be (N, C) and target must be (N,)")
        k = min(self.k, scores.shape[1])
        topk = torch.topk(scores, k=k, dim=1).indices
        hits = (topk == target.long().unsqueeze(1)).any(dim=1)
        return hits.float().mean().item()


class BLEUScore:
    """Corpus BLEU score for tokenized generated text.

    Inputs are batches of token sequences. A sequence can be a list of tokens,
    a whitespace-separated string, or a 1D tensor of token ids.
    """

    def __init__(
        self,
        max_n: int = 4,
        smooth: float = 1.0,
        lowercase: bool = False,
    ) -> None:
        if max_n <= 0:
            raise ValueError("max_n must be positive")
        self.max_n = max_n
        self.smooth = smooth
        self.lowercase = lowercase

    def __call__(self, pred: Sequence[Any], target: Sequence[Any]) -> float:
        pred_sequences = _normalize_sequence_batch(pred, self.lowercase)
        target_sequences = _normalize_sequence_batch(target, self.lowercase)
        if len(pred_sequences) != len(target_sequences):
            raise ValueError(
                "pred and target must contain the same number of sequences"
            )
        if not pred_sequences:
            return 0.0

        precisions = []
        for ngram_size in range(1, self.max_n + 1):
            overlap = 0.0
            total = 0.0
            for pred_tokens, target_tokens in zip(pred_sequences, target_sequences):
                pred_counts = _ngram_counts(pred_tokens, ngram_size)
                target_counts = _ngram_counts(target_tokens, ngram_size)
                overlap += sum(
                    min(count, target_counts.get(ngram, 0))
                    for ngram, count in pred_counts.items()
                )
                total += sum(pred_counts.values())
            precisions.append((overlap + self.smooth) / (total + self.smooth))

        pred_length = sum(len(tokens) for tokens in pred_sequences)
        target_length = sum(len(tokens) for tokens in target_sequences)
        if pred_length == 0:
            return 0.0
        brevity = (
            1.0
            if pred_length > target_length
            else math.exp(1.0 - target_length / pred_length)
        )
        score = brevity * math.exp(sum(math.log(p) for p in precisions) / self.max_n)
        return float(score)


class ROUGELScore:
    """Corpus ROUGE-L F1 score for tokenized generated text."""

    def __init__(self, beta: float = 1.2, lowercase: bool = False) -> None:
        self.beta = beta
        self.lowercase = lowercase

    def __call__(self, pred: Sequence[Any], target: Sequence[Any]) -> float:
        pred_sequences = _normalize_sequence_batch(pred, self.lowercase)
        target_sequences = _normalize_sequence_batch(target, self.lowercase)
        if len(pred_sequences) != len(target_sequences):
            raise ValueError(
                "pred and target must contain the same number of sequences"
            )
        if not pred_sequences:
            return 0.0
        scores = [
            _rouge_l_f1(pred_tokens, target_tokens, self.beta)
            for pred_tokens, target_tokens in zip(pred_sequences, target_sequences)
        ]
        return sum(scores) / len(scores)


def _normalize_sequence_batch(batch: Sequence[Any], lowercase: bool) -> list[list[str]]:
    normalized = []
    for item in batch:
        if isinstance(item, str):
            tokens = item.split()
        elif isinstance(item, torch.Tensor):
            tokens = [str(token.item()) for token in item.reshape(-1)]
        else:
            tokens = [str(token) for token in item]
        if lowercase:
            tokens = [token.lower() for token in tokens]
        normalized.append(tokens)
    return normalized


def _ngram_counts(tokens: Sequence[str], ngram_size: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if len(tokens) < ngram_size:
        return counts
    for start in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[start : start + ngram_size])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _rouge_l_f1(
    pred_tokens: Sequence[str], target_tokens: Sequence[str], beta: float
) -> float:
    if not pred_tokens or not target_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, target_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(target_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    beta_sq = beta**2
    return ((1.0 + beta_sq) * precision * recall) / (recall + beta_sq * precision)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
