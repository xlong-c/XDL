"""Linear probe helpers for layer-wise representation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError(f"Expected features with ndim >= 2, got shape {tuple(features.shape)}")
    if features.ndim == 2:
        return features
    return features.reshape(features.shape[0], -1)


@dataclass
class LinearProbe:
    """A fitted linear classifier over frozen features."""

    model: nn.Linear
    feature_dim: int
    num_classes: int
    train_loss: float
    epochs: int

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        flattened = _flatten_features(features).to(
            device=self.model.weight.device,
            dtype=self.model.weight.dtype,
        )
        return self.model(flattened)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict_logits(features).argmax(dim=-1)

    def score(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        predictions = self.predict(features)
        accuracy = (predictions.cpu() == labels.detach().cpu()).float().mean()
        return float(accuracy.item())


def fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: Optional[int] = None,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    epochs: int = 200,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> LinearProbe:
    """Fit a linear probe on frozen features with cross-entropy."""

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Features and labels must have the same batch size, got {features.shape[0]} and {labels.shape[0]}"
        )
    flattened = _flatten_features(features).detach().float()
    targets = labels.detach().long()
    inferred_classes = int(targets.max().item()) + 1 if targets.numel() else 0
    class_count = num_classes if num_classes is not None else inferred_classes
    if class_count <= 1:
        raise ValueError(f"Linear probe requires at least 2 classes, got {class_count}")

    train_device = device or torch.device("cpu")
    model = nn.Linear(flattened.shape[1], class_count).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(flattened, targets)
    effective_batch_size = batch_size or min(256, len(dataset))
    loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)

    final_loss = 0.0
    model.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(train_device)
            batch_labels = batch_labels.to(train_device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

    return LinearProbe(
        model=model.eval(),
        feature_dim=int(flattened.shape[1]),
        num_classes=class_count,
        train_loss=final_loss,
        epochs=epochs,
    )


def score_linear_probe(
    probe: LinearProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Convenience wrapper for probe accuracy."""

    return probe.score(features, labels)


__all__ = [
    "LinearProbe",
    "fit_linear_probe",
    "score_linear_probe",
]
