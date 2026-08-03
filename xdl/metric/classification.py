"""Classification, regression, and multilabel metrics."""

from __future__ import annotations

import torch

from xdl.metric._utils import (
    _confusion_matrix_from_labels,
    _multilabel_score,
    _multilabel_stats,
    _precision_recall_f1,
    _safe_divide,
    _to_label_predictions
)


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

