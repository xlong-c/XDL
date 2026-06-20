"""Sequence and language-modeling losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classification_loss import _validate_reduction


class MaskedCrossEntropyLoss(nn.Module):
    """Cross entropy for token sequences with optional masks.

    Args:
        ignore_index: Target value ignored by the loss.
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.
        label_smoothing: Uniform label smoothing passed to PyTorch cross entropy.

    Shape:
        ``logits`` is ``(..., C)`` and ``target`` is ``(...)``.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.dim() != target.dim() + 1:
            raise ValueError(
                f"logits must have one more dimension than target: {logits.shape} vs {target.shape}"
            )

        num_classes = logits.shape[-1]
        flat_logits = logits.reshape(-1, num_classes)
        flat_target = target.reshape(-1).long()
        flat_loss = F.cross_entropy(
            flat_logits,
            flat_target,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        valid = flat_target != self.ignore_index
        if mask is not None:
            if mask.shape != target.shape:
                raise ValueError(
                    f"mask shape must match target: {mask.shape} vs {target.shape}"
                )
            valid = valid & mask.reshape(-1).bool()

        if self.reduction == "none":
            return torch.where(valid, flat_loss, torch.zeros_like(flat_loss)).reshape(
                target.shape
            )
        if self.reduction == "sum":
            return flat_loss[valid].sum()
        if not torch.any(valid):
            return flat_loss.sum() * 0.0
        return flat_loss[valid].mean()


class SequenceCrossEntropyLoss(MaskedCrossEntropyLoss):
    """Alias-style sequence cross entropy for token classification tasks."""


class TokenClassificationLoss(MaskedCrossEntropyLoss):
    """Named variant for NLP sequence labeling."""


class CausalLanguageModelingLoss(nn.Module):
    """Next-token prediction loss for causal language models.

    Args:
        ignore_index: Target value ignored by the loss.
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.
        label_smoothing: Uniform label smoothing.

    Shape:
        ``logits`` is ``(B, T, V)`` and ``labels`` is ``(B, T)``. The loss shifts
        logits left and labels right internally.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_loss = MaskedCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3 or labels.dim() != 2:
            raise ValueError("logits must be (B, T, V) and labels must be (B, T)")
        if logits.shape[:2] != labels.shape:
            raise ValueError(
                f"logits and labels sequence dimensions differ: {logits.shape[:2]} vs {labels.shape}"
            )
        if logits.shape[1] < 2:
            raise ValueError("sequence length must be at least 2")
        return self.token_loss(
            logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
        )
