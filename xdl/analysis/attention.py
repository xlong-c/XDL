"""Attention rollout helpers and model adapters for transformer interpretation."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn


def _extract_attentions_from_output(output: Any) -> Optional[tuple[torch.Tensor, ...]]:
    if isinstance(output, Mapping):
        attentions = output.get("attentions")
        if attentions is not None:
            collected = tuple(attentions)
            return collected if collected else None
    if hasattr(output, "attentions") and getattr(output, "attentions") is not None:
        collected = tuple(getattr(output, "attentions"))
        return collected if collected else None
    return None


class AttentionCapture:
    """Capture attention maps from supported transformer-style models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def capture(
        self,
        *forward_args: Any,
        forward_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> tuple[torch.Tensor, ...]:
        kwargs = dict(forward_kwargs or {})

        direct_attentions = self._capture_from_model_output(*forward_args, forward_kwargs=kwargs)
        if direct_attentions is not None:
            return direct_attentions

        hook_attentions = self._capture_via_hooks(*forward_args, forward_kwargs=kwargs)
        if hook_attentions:
            return hook_attentions

        raise ValueError(
            "Could not capture attentions from model output or registered attention modules"
        )

    def _capture_from_model_output(
        self,
        *forward_args: Any,
        forward_kwargs: Mapping[str, Any],
    ) -> Optional[tuple[torch.Tensor, ...]]:
        kwargs = dict(forward_kwargs)
        output = None
        try:
            output = self.model(*forward_args, **kwargs)
        except TypeError:
            if "output_attentions" in kwargs:
                raise

        attentions = _extract_attentions_from_output(output)
        if attentions is not None:
            return tuple(att.detach() for att in attentions)

        kwargs = dict(forward_kwargs)
        kwargs["output_attentions"] = True
        try:
            output = self.model(*forward_args, **kwargs)
        except TypeError:
            return None
        attentions = _extract_attentions_from_output(output)
        if attentions is None:
            return None
        return tuple(att.detach() for att in attentions)

    def _capture_via_hooks(
        self,
        *forward_args: Any,
        forward_kwargs: Mapping[str, Any],
    ) -> tuple[torch.Tensor, ...]:
        captures: list[torch.Tensor] = []
        handles: list[Any] = []
        try:
            for module in self.model.modules():
                if _looks_like_attention_module(module):
                    handle = module.register_forward_hook(
                        lambda module_ref, _inputs, output, bucket=captures: _append_attention_output(
                            module_ref,
                            output,
                            bucket,
                        )
                    )
                    handles.append(handle)
            if not handles:
                return ()
            self.model(*forward_args, **dict(forward_kwargs))
            return tuple(tensor.detach() for tensor in captures)
        finally:
            while handles:
                handles.pop().remove()


def _looks_like_attention_module(module: nn.Module) -> bool:
    module_name = module.__class__.__name__.lower()
    if module_name in {"multiheadattention", "clipattention", "vitattention"}:
        return True
    return hasattr(module, "last_attention_weights")


def _append_attention_output(module: nn.Module, output: Any, bucket: list[torch.Tensor]) -> None:
    attention_weights = getattr(module, "last_attention_weights", None)
    if isinstance(attention_weights, torch.Tensor) and attention_weights.ndim in (3, 4):
        bucket.append(attention_weights)
        return
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim in (3, 4):
                bucket.append(item)
                return
    if isinstance(output, torch.Tensor) and output.ndim in (3, 4):
        bucket.append(output)


def capture_attention_maps(
    model: nn.Module,
    *forward_args: Any,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[torch.Tensor, ...]:
    """Capture attention tensors from a supported ViT/CLIP style model."""

    return AttentionCapture(model).capture(*forward_args, forward_kwargs=forward_kwargs)


def attention_rollout_for_model(
    model: nn.Module,
    *forward_args: Any,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
    average_heads: bool = True,
    add_residual: bool = True,
    normalize_rows: bool = True,
) -> torch.Tensor:
    """Capture attention tensors from a model and return the rollout matrix."""

    attentions = capture_attention_maps(model, *forward_args, forward_kwargs=forward_kwargs)
    return attention_rollout(
        attentions,
        average_heads=average_heads,
        add_residual=add_residual,
        normalize_rows=normalize_rows,
    )


def attention_rollout(
    attentions: Sequence[torch.Tensor],
    *,
    average_heads: bool = True,
    add_residual: bool = True,
    normalize_rows: bool = True,
) -> torch.Tensor:
    """Aggregate a list of self-attention maps into a rollout matrix.

    Args:
        attentions: Sequence of attention tensors with shape
            [batch, heads, tokens, tokens] or [batch, tokens, tokens].
        average_heads: Whether to average across heads before composing.
        add_residual: Whether to add identity before normalization.
        normalize_rows: Whether to normalize each row to sum to 1.
    """

    if not attentions:
        raise ValueError("attention_rollout requires at least one attention tensor")

    rollout: torch.Tensor | None = None
    for attention in attentions:
        if attention.ndim == 4:
            matrix = attention.mean(dim=1) if average_heads else attention.sum(dim=1)
        elif attention.ndim == 3:
            matrix = attention
        else:
            raise ValueError(
                f"Attention tensor must have shape [batch, heads, tokens, tokens] or [batch, tokens, tokens], got {tuple(attention.shape)}"
            )
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"Attention matrix must be square, got {tuple(matrix.shape)}")

        current = matrix.detach().float()
        if add_residual:
            identity = torch.eye(current.shape[-1], device=current.device, dtype=current.dtype)
            current = current + identity.unsqueeze(0)
        if normalize_rows:
            current = current / current.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        rollout = current if rollout is None else current @ rollout
    assert rollout is not None
    return rollout


__all__ = [
    "AttentionCapture",
    "attention_rollout",
    "attention_rollout_for_model",
    "capture_attention_maps",
]
