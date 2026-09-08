"""Concept probes and TCAV-style analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError(f"Expected features with ndim >= 2, got shape {tuple(features.shape)}")
    if features.ndim == 2:
        return features
    return features.reshape(features.shape[0], -1)


@dataclass
class ConceptProbe:
    """Binary linear concept direction."""

    weight: torch.Tensor
    bias: torch.Tensor
    train_loss: float
    epochs: int

    @property
    def direction(self) -> torch.Tensor:
        return self.weight.detach().reshape(-1)

    def score_samples(self, features: torch.Tensor) -> torch.Tensor:
        flattened = _flatten_features(features).to(dtype=self.weight.dtype, device=self.weight.device)
        logits = flattened @ self.weight.view(-1, 1) + self.bias
        return logits.view(-1)

    def predict(self, features: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        return (self.score_samples(features) > threshold).long()

    def accuracy(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        predictions = self.predict(features).cpu()
        accuracy = (predictions == labels.detach().long().cpu()).float().mean()
        return float(accuracy.item())


def fit_concept_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    epochs: int = 200,
    device: Optional[torch.device] = None,
) -> ConceptProbe:
    """Fit a binary linear probe that separates concept positives and negatives."""

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Features and labels must have the same batch size, got {features.shape[0]} and {labels.shape[0]}"
        )
    flattened = _flatten_features(features).detach().float()
    targets = labels.detach().float().view(-1)
    unique_values = set(int(value) for value in targets.unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Concept probe expects binary labels in {{0,1}}, got {sorted(unique_values)}")

    train_device = device or torch.device("cpu")
    weight = torch.nn.Parameter(torch.zeros(flattened.shape[1], device=train_device))
    bias = torch.nn.Parameter(torch.zeros(1, device=train_device))
    optimizer = torch.optim.AdamW([weight, bias], lr=lr, weight_decay=weight_decay)

    batch_features = flattened.to(train_device)
    batch_labels = targets.to(train_device)
    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = batch_features @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    return ConceptProbe(
        weight=weight.detach(),
        bias=bias.detach(),
        train_loss=final_loss,
        epochs=epochs,
    )


def concept_activation_vector(probe: ConceptProbe) -> torch.Tensor:
    """Return the learned concept direction."""

    return probe.direction


def tcav_score(
    gradients: torch.Tensor,
    concept_direction: torch.Tensor,
) -> float:
    """Compute the fraction of samples with positive directional derivative."""

    flattened_gradients = _flatten_features(gradients).detach().float()
    direction = concept_direction.detach().float().reshape(-1)
    if flattened_gradients.shape[1] != direction.numel():
        raise ValueError(
            f"Gradient feature dim {flattened_gradients.shape[1]} does not match concept direction dim {direction.numel()}"
        )
    directional_derivative = flattened_gradients @ direction
    return float((directional_derivative > 0).float().mean().item())


def tcav_from_probe(
    gradients: torch.Tensor,
    probe: ConceptProbe,
) -> float:
    """Compute a TCAV score directly from a fitted concept probe."""

    return tcav_score(gradients, probe.direction)


def directional_derivative(
    gradients: torch.Tensor,
    concept_direction: torch.Tensor,
) -> torch.Tensor:
    """Project gradients onto a concept direction sample-wise."""

    flattened_gradients = _flatten_features(gradients).detach().float()
    direction = concept_direction.detach().float().reshape(-1)
    if flattened_gradients.shape[1] != direction.numel():
        raise ValueError(
            f"Gradient feature dim {flattened_gradients.shape[1]} does not match concept direction dim {direction.numel()}"
        )
    return flattened_gradients @ direction


def compute_module_tcav(
    model: nn.Module,
    module_name: str,
    concept_direction: torch.Tensor,
    *forward_args: Any,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
    target_index: Optional[int | torch.Tensor] = None,
    output_selector: Optional[Callable[[Any], torch.Tensor]] = None,
) -> tuple[float, torch.Tensor]:
    """Compute TCAV score and directional derivatives for one module on one batch."""

    kwargs = dict(forward_kwargs or {})
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"Module not found: {module_name}")

    target_module = modules[module_name]
    captured: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("TCAV target module must output a Tensor")
        captured["value"] = tensor

    def backward_hook(
        _module: nn.Module,
        _grad_input: Any,
        grad_output: Any,
    ) -> None:
        grad_tensor = grad_output[0]
        if isinstance(grad_tensor, torch.Tensor):
            gradients["value"] = grad_tensor

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    was_training = model.training
    model.eval()
    try:
        outputs = model(*forward_args, **kwargs)
        logits = output_selector(outputs) if output_selector is not None else _default_output_selector(outputs)
        if logits.ndim != 2:
            raise ValueError(f"TCAV expects logits with shape [batch, classes], got {tuple(logits.shape)}")

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

        if "value" not in captured or "value" not in gradients:
            raise RuntimeError("TCAV hooks did not capture activations and gradients")

        derivatives = directional_derivative(gradients["value"], concept_direction)
        return float((derivatives > 0).float().mean().item()), derivatives.detach()
    finally:
        forward_handle.remove()
        backward_handle.remove()
        if was_training:
            model.train()


def _default_output_selector(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first
    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits
    logits = getattr(outputs, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise TypeError("Could not resolve classifier logits from model outputs")


__all__ = [
    "ConceptProbe",
    "concept_activation_vector",
    "compute_module_tcav",
    "directional_derivative",
    "fit_concept_probe",
    "tcav_from_probe",
    "tcav_score",
]
