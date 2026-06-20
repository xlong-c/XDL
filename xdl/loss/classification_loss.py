"""Classification losses for hard labels, soft targets and multi-label tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    _validate_reduction(reduction)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with uniform label smoothing.

    Args:
        smoothing: Probability mass mixed into the uniform class distribution.
        ignore_index: Target value ignored by the loss.
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.

    Shape:
        ``logits`` is ``(N, C, ...)`` and ``target`` is ``(N, ...)``.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        _validate_reduction(reduction)
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() < 2:
            raise ValueError("logits must have shape (N, C, ...)")

        num_classes = logits.shape[1]
        target_shape = target.shape
        log_probs = F.log_softmax(logits, dim=1)
        flat_log_probs = log_probs.movedim(1, -1).reshape(-1, num_classes)
        flat_target = target.reshape(-1).long()
        valid = flat_target != self.ignore_index

        if not torch.any(valid):
            empty_loss = flat_log_probs.sum() * 0.0
            if self.reduction == "none":
                return target.new_zeros(target_shape, dtype=logits.dtype)
            return empty_loss

        valid_log_probs = flat_log_probs[valid]
        valid_target = flat_target[valid]
        nll_loss = -valid_log_probs.gather(1, valid_target.unsqueeze(1)).squeeze(1)
        smooth_loss = -valid_log_probs.mean(dim=1)
        valid_loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        if self.reduction == "none":
            flat_loss = flat_log_probs.new_zeros(flat_target.shape)
            flat_loss[valid] = valid_loss
            return flat_loss.reshape(target_shape)
        return _reduce_loss(valid_loss, self.reduction)


class SoftTargetCrossEntropy(nn.Module):
    """Cross entropy for probability targets from distillation, MixUp or CutMix.

    Args:
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.

    Shape:
        ``logits`` and ``target`` have the same shape ``(N, C, ...)``. The class
        dimension is expected at index 1, matching PyTorch classification losses.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"logits and target must have the same shape: {logits.shape} vs {target.shape}"
            )
        if logits.dim() < 2:
            raise ValueError("logits must have shape (N, C, ...)")

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(target.to(dtype=logits.dtype) * log_probs).sum(dim=1)
        return _reduce_loss(loss, self.reduction)


class AsymmetricLoss(nn.Module):
    """Asymmetric focal loss for multi-label classification.

    This is commonly used for imbalanced image tagging and multi-label NLP
    classification. Inputs are logits and targets are multi-hot tensors.

    Args:
        gamma_neg: Focusing factor for negative labels.
        gamma_pos: Focusing factor for positive labels.
        clip: Optional probability clipping added to negative probabilities.
        eps: Numerical stability epsilon.
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"logits and target must have the same shape: {logits.shape} vs {target.shape}"
            )

        target = target.to(dtype=logits.dtype)
        prob_pos = torch.sigmoid(logits)
        prob_neg = 1.0 - prob_pos
        if self.clip > 0.0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        log_pos = torch.log(prob_pos.clamp(min=self.eps))
        log_neg = torch.log(prob_neg.clamp(min=self.eps))
        loss = target * log_pos + (1.0 - target) * log_neg

        pt = prob_pos * target + prob_neg * (1.0 - target)
        gamma = self.gamma_pos * target + self.gamma_neg * (1.0 - target)
        focal_weight = (1.0 - pt).clamp(min=0.0).pow(gamma)
        loss = -loss * focal_weight
        return _reduce_loss(loss, self.reduction)
