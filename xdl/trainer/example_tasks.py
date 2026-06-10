"""Minimal CoreModel tasks for official config examples."""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn
from xdl.metric import Accuracy, DiceCoefficient, IoU, MeanAbsoluteError, MeanSquaredError

from .coreModel import CoreModel


class TinySegmentationTask(CoreModel):
    """A tiny segmentation task that consumes dict batches from manifest datasets."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        hidden_channels: int = 16,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lr = float(lr)
        self.metrics: List[Any] = [
            IoU(num_classes=int(num_classes)),
            DiceCoefficient(num_classes=int(num_classes)),
        ]
        self.model = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(hidden_channels), int(num_classes), kernel_size=1),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        image = batch["image"]
        mask = batch["mask"].long()
        logits = self.forward(image)
        loss = self.loss_fn(logits, mask)

        stepped = self.manual_optimization_step(
            loss,
            optimizer=self.optimizers[0],
            model=self.model,
        )
        self.log("loss", loss, prefix="train")
        self._log_metrics(logits, mask, prefix="train")
        self.log("optimizer_step", float(stepped), prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        image = batch["image"]
        mask = batch["mask"].long()
        logits = self.forward(image)
        loss = self.loss_fn(logits, mask)

        self.log("loss", loss, prefix="val")
        self._log_metrics(logits, mask, prefix="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _log_metrics(self, logits: torch.Tensor, mask: torch.Tensor, *, prefix: str) -> None:
        for metric in self.metrics:
            metric_name = type(metric).__name__
            self.log(metric_name, metric(logits, mask), prefix=prefix)


class TinyDetectionTask(CoreModel):
    """A tiny detection task that consumes DetectionCollate batches."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        hidden_channels: int = 16,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lr = float(lr)
        self.num_classes = int(num_classes)
        self.metrics: List[Any] = [Accuracy(num_classes=self.num_classes)]
        self.encoder = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.cls_head = nn.Linear(int(hidden_channels), self.num_classes)
        self.box_head = nn.Linear(int(hidden_channels), 4)
        self.cls_loss = nn.CrossEntropyLoss()
        self.box_loss = nn.SmoothL1Loss()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(image)
        return self.cls_head(features), self.box_head(features)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        image = batch["image"]
        label_targets, box_targets = self._first_targets(batch)
        class_logits, box_pred = self.forward(image)
        loss = self.cls_loss(class_logits, label_targets) + self.box_loss(box_pred, box_targets)

        self.manual_optimization_step(
            loss,
            optimizer=self.optimizers[0],
            model=self,
        )
        self.log("loss", loss, prefix="train")
        self._log_metrics(class_logits, label_targets, prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        image = batch["image"]
        label_targets, box_targets = self._first_targets(batch)
        class_logits, box_pred = self.forward(image)
        loss = self.cls_loss(class_logits, label_targets) + self.box_loss(box_pred, box_targets)

        self.log("loss", loss, prefix="val")
        self._log_metrics(class_logits, label_targets, prefix="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _first_targets(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        labels = batch["labels"]
        boxes = batch["boxes"]
        label_targets = torch.stack(
            [
                item[0].long() if item.numel() > 0 else torch.tensor(0, device=self.device)
                for item in labels
            ]
        )
        box_targets = torch.stack(
            [
                item[0].float() if item.numel() > 0 else torch.zeros(4, device=self.device)
                for item in boxes
            ]
        )
        return label_targets, box_targets

    def _log_metrics(
        self,
        class_logits: torch.Tensor,
        label_targets: torch.Tensor,
        *,
        prefix: str,
    ) -> None:
        for metric in self.metrics:
            self.log(type(metric).__name__, metric(class_logits, label_targets), prefix=prefix)


class TinyRegressionTask(CoreModel):
    """A tiny regression task that consumes tuple batches from manifest regression datasets."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lr = float(lr)
        self.metrics: List[Any] = [MeanAbsoluteError(), MeanSquaredError()]
        self.encoder = nn.Sequential(
            nn.Conv2d(int(input_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(hidden_channels), 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image).squeeze(-1)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        image, target = batch
        target_tensor = target.float()
        pred = self.forward(image)
        loss = self.loss_fn(pred, target_tensor)

        self.manual_optimization_step(
            loss,
            optimizer=self.optimizers[0],
            model=self,
        )
        self.log("loss", loss, prefix="train")
        self._log_metrics(pred, target_tensor, prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        image, target = batch
        target_tensor = target.float()
        pred = self.forward(image)
        loss = self.loss_fn(pred, target_tensor)

        self.log("loss", loss, prefix="val")
        self._log_metrics(pred, target_tensor, prefix="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _log_metrics(self, pred: torch.Tensor, target: torch.Tensor, *, prefix: str) -> None:
        for metric in self.metrics:
            self.log(type(metric).__name__, metric(pred, target), prefix=prefix)


class TinyPairClassificationTask(CoreModel):
    """A tiny pair task that predicts whether two images belong together."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        embedding_dim: int = 16,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lr = float(lr)
        self.metrics: List[Any] = [Accuracy(num_classes=2)]
        self.encoder = nn.Sequential(
            nn.Conv2d(int(input_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(hidden_channels), int(embedding_dim)),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(int(embedding_dim) * 3, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        feat_a = self.encode(image_a)
        feat_b = self.encode(image_b)
        pair_features = torch.cat([feat_a, feat_b, torch.abs(feat_a - feat_b)], dim=1)
        return self.classifier(pair_features)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        image_a = batch["image_a"]
        image_b = batch["image_b"]
        labels = torch.as_tensor(batch["label"], device=self.device).long()
        logits = self.forward(image_a, image_b)
        loss = self.loss_fn(logits, labels)

        self.manual_optimization_step(
            loss,
            optimizer=self.optimizers[0],
            model=self,
        )
        self.log("loss", loss, prefix="train")
        self._log_metrics(logits, labels, prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        image_a = batch["image_a"]
        image_b = batch["image_b"]
        labels = torch.as_tensor(batch["label"], device=self.device).long()
        logits = self.forward(image_a, image_b)
        loss = self.loss_fn(logits, labels)

        self.log("loss", loss, prefix="val")
        self._log_metrics(logits, labels, prefix="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _log_metrics(self, logits: torch.Tensor, labels: torch.Tensor, *, prefix: str) -> None:
        for metric in self.metrics:
            self.log(type(metric).__name__, metric(logits, labels), prefix=prefix)
