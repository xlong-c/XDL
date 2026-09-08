"""Grad-CAM helpers for convolutional and patch-based feature maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn


def _resolve_target_module(model: nn.Module, target_module: str | nn.Module) -> nn.Module:
    if isinstance(target_module, nn.Module):
        return target_module
    modules = dict(model.named_modules())
    if target_module not in modules:
        raise KeyError(f"Module not found: {target_module}")
    return modules[target_module]


def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    flat = heatmap.flatten(start_dim=1)
    min_values = flat.min(dim=1).values.view(-1, 1, 1)
    max_values = flat.max(dim=1).values.view(-1, 1, 1)
    denom = (max_values - min_values).clamp_min(1e-12)
    return (heatmap - min_values) / denom


@dataclass
class GradCAM:
    """Result object for one Grad-CAM run."""

    heatmap: torch.Tensor
    activations: torch.Tensor
    gradients: torch.Tensor
    logits: torch.Tensor
    target_indices: torch.Tensor


def compute_grad_cam(
    model: nn.Module,
    target_module: str | nn.Module,
    *forward_args: Any,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
    target_index: Optional[int | torch.Tensor] = None,
) -> GradCAM:
    """Compute Grad-CAM for a target module and classifier output."""

    kwargs = dict(forward_kwargs or {})
    target_layer = _resolve_target_module(model, target_module)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    forward_handle = None
    backward_handle = None

    def forward_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Grad-CAM target module must output a Tensor")
        activations["value"] = tensor

    def backward_hook(
        _module: nn.Module,
        _grad_input: Any,
        grad_output: Any,
    ) -> None:
        grad_tensor = grad_output[0]
        if isinstance(grad_tensor, torch.Tensor):
            gradients["value"] = grad_tensor

    was_training = model.training
    model.eval()
    try:
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        outputs = model(*forward_args, **kwargs)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Grad-CAM expects model outputs to be a Tensor or tuple with Tensor first")
        if logits.ndim != 2:
            raise ValueError(f"Grad-CAM expects classifier logits with shape [batch, classes], got {tuple(logits.shape)}")

        if target_index is None:
            target_indices = logits.argmax(dim=-1)
        elif isinstance(target_index, int):
            target_indices = torch.full(
                (logits.shape[0],),
                fill_value=target_index,
                device=logits.device,
                dtype=torch.long,
            )
        else:
            target_indices = target_index.to(device=logits.device, dtype=torch.long)
            if target_indices.shape != (logits.shape[0],):
                raise ValueError(
                    f"target_index tensor must have shape ({logits.shape[0]},), got {tuple(target_indices.shape)}"
                )

        score = logits.gather(1, target_indices.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

        activation_tensor = activations["value"]
        gradient_tensor = gradients["value"]
        if activation_tensor.ndim != 4:
            raise ValueError(
                f"Grad-CAM currently supports 4-D feature maps [batch, channels, height, width], got {tuple(activation_tensor.shape)}"
            )

        weights = gradient_tensor.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation_tensor).sum(dim=1))
        normalized = _normalize_heatmap(cam.detach())
        return GradCAM(
            heatmap=normalized,
            activations=activation_tensor.detach(),
            gradients=gradient_tensor.detach(),
            logits=logits.detach(),
            target_indices=target_indices.detach(),
        )
    finally:
        if forward_handle is not None:
            forward_handle.remove()
        if backward_handle is not None:
            backward_handle.remove()
        if was_training:
            model.train()


__all__ = [
    "GradCAM",
    "compute_grad_cam",
]
