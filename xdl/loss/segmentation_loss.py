"""Segmentation losses for masks and dense prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classification_loss import _validate_reduction


def _spatial_sum_dims(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(dim for dim in range(tensor.dim()) if dim != 1)


def _prepare_dense_targets(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    from_logits: bool,
    ignore_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() < 3:
        raise ValueError("segmentation logits must have shape (N, C, ...)")

    num_classes = logits.shape[1]
    if num_classes == 1:
        pred = torch.sigmoid(logits) if from_logits else logits
        target_dense = target
        if target_dense.dim() == logits.dim() - 1:
            target_dense = target_dense.unsqueeze(1)
        if target_dense.shape != pred.shape:
            raise ValueError(
                f"binary target shape must match logits without/with channel: {target.shape}"
            )
        target_float = target_dense.to(dtype=logits.dtype)
        if ignore_index is not None:
            valid = target_dense != ignore_index
            pred = pred * valid.to(dtype=logits.dtype)
            target_float = torch.where(
                valid, target_float, torch.zeros_like(target_float)
            )
        return pred, target_float

    pred = torch.softmax(logits, dim=1) if from_logits else logits
    if target.dim() == logits.dim():
        if target.shape != logits.shape:
            raise ValueError(
                f"one-hot target shape must match logits: {target.shape} vs {logits.shape}"
            )
        return pred, target.to(dtype=logits.dtype)

    if target.dim() != logits.dim() - 1:
        raise ValueError(
            f"target must have shape (N, ...) or (N, C, ...), got {target.shape}"
        )

    target_long = target.long()
    if ignore_index is None:
        valid = torch.ones_like(target_long, dtype=torch.bool)
        safe_target = target_long
    else:
        valid = target_long != ignore_index
        safe_target = torch.where(valid, target_long, torch.zeros_like(target_long))

    target_onehot = F.one_hot(safe_target, num_classes=num_classes)
    target_onehot = target_onehot.movedim(-1, 1).to(dtype=logits.dtype)
    valid_float = valid.unsqueeze(1).to(dtype=logits.dtype)
    return pred * valid_float, target_onehot * valid_float


def _aggregate_score(
    score_per_class: torch.Tensor,
    average: str,
) -> torch.Tensor:
    if average == "macro":
        return score_per_class.mean()
    if average == "none":
        return score_per_class
    raise ValueError("average must be one of: 'macro', 'micro', 'none'")


class JaccardLoss(nn.Module):
    """Soft Jaccard/IoU loss for binary or multi-class segmentation.

    Args:
        smooth: Smoothing term added to numerator and denominator.
        average: ``'macro'``, ``'micro'`` or ``'none'``.
        from_logits: Whether to apply sigmoid/softmax to predictions.
        ignore_index: Label value ignored for integer mask targets.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        average: str = "macro",
        from_logits: bool = True,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        if average not in {"macro", "micro", "none"}:
            raise ValueError("average must be one of: 'macro', 'micro', 'none'")
        self.smooth = smooth
        self.average = average
        self.from_logits = from_logits
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target_onehot = _prepare_dense_targets(
            logits,
            target,
            from_logits=self.from_logits,
            ignore_index=self.ignore_index,
        )
        dims = _spatial_sum_dims(pred)
        intersection = (pred * target_onehot).sum(dim=dims)
        union = pred.sum(dim=dims) + target_onehot.sum(dim=dims) - intersection

        if self.average == "micro":
            score = (intersection.sum() + self.smooth) / (union.sum() + self.smooth)
            return 1.0 - score

        score_per_class = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - _aggregate_score(score_per_class, self.average)


class TverskyLoss(nn.Module):
    """Tversky loss for imbalanced segmentation masks.

    Args:
        alpha: False positive penalty.
        beta: False negative penalty.
        smooth: Smoothing term added to numerator and denominator.
        average: ``'macro'``, ``'micro'`` or ``'none'``.
        from_logits: Whether to apply sigmoid/softmax to predictions.
        ignore_index: Label value ignored for integer mask targets.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        average: str = "macro",
        from_logits: bool = True,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        if average not in {"macro", "micro", "none"}:
            raise ValueError("average must be one of: 'macro', 'micro', 'none'")
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.average = average
        self.from_logits = from_logits
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target_onehot = _prepare_dense_targets(
            logits,
            target,
            from_logits=self.from_logits,
            ignore_index=self.ignore_index,
        )
        dims = _spatial_sum_dims(pred)
        true_pos = (pred * target_onehot).sum(dim=dims)
        false_pos = (pred * (1.0 - target_onehot)).sum(dim=dims)
        false_neg = ((1.0 - pred) * target_onehot).sum(dim=dims)

        if self.average == "micro":
            true_pos = true_pos.sum()
            false_pos = false_pos.sum()
            false_neg = false_neg.sum()
            score = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
            )
            return 1.0 - score

        score_per_class = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        return 1.0 - _aggregate_score(score_per_class, self.average)


class FocalTverskyLoss(TverskyLoss):
    """Focal Tversky loss, useful for small foreground objects.

    Args:
        gamma: Focal exponent applied to the Tversky loss.
        Other arguments match :class:`TverskyLoss`.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        smooth: float = 1.0,
        average: str = "macro",
        from_logits: bool = True,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            average=average,
            from_logits=from_logits,
            ignore_index=ignore_index,
        )
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky_loss = super().forward(logits, target)
        return tversky_loss.pow(self.gamma)


class DiceCrossEntropyLoss(nn.Module):
    """Weighted sum of Dice loss and cross entropy for segmentation.

    Args:
        dice_weight: Weight applied to Dice loss.
        ce_weight: Weight applied to cross entropy or binary cross entropy.
        smooth: Dice smoothing term.
        ignore_index: Label value ignored by cross entropy and Dice.
        reduction: Cross entropy reduction, ``'mean'`` or ``'sum'``.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction == "none":
            raise ValueError("DiceCrossEntropyLoss does not support reduction='none'")
        _validate_reduction(reduction)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dice = TverskyLoss(
            alpha=0.5,
            beta=0.5,
            smooth=smooth,
            average="macro",
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            ce_target = target
            if ce_target.dim() == logits.dim() - 1:
                ce_target = ce_target.unsqueeze(1)
            valid = ce_target != self.ignore_index
            if torch.any(valid):
                ce_loss = F.binary_cross_entropy_with_logits(
                    logits[valid],
                    ce_target.to(dtype=logits.dtype)[valid],
                    reduction=self.reduction,
                )
            else:
                ce_loss = logits.sum() * 0.0
        else:
            ce_loss = F.cross_entropy(
                logits,
                target.long(),
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
        dice_loss = self.dice(logits, target)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss
