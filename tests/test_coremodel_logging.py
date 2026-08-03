from typing import Any

import torch

from xdl.trainer import CoreModel


class LoggingModel(CoreModel):
    def configure_optimizers(self) -> None:
        return None

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx


def test_log_prefix_participates_in_metric_name() -> None:
    model = LoggingModel()

    model.log("loss", torch.tensor(1.25), prefix="train")
    model.log("accuracy", 0.75, prefix="val")

    assert model.current_metrics == {
        "train_loss": 1.25,
        "val_accuracy": 0.75,
    }
    assert model._step_metrics.get("train_loss") == [1.25]
    assert model._step_metrics.get("val_accuracy") == [0.75]


def test_log_prefix_does_not_duplicate_existing_prefix() -> None:
    model = LoggingModel()

    model.log("train_loss", 1.0, prefix="train")
    model.log("train/loss", 2.0, prefix="train")

    assert model.current_metrics == {
        "train_loss": 1.0,
        "train/loss": 2.0,
    }


def test_log_metrics_applies_prefix_to_all_entries() -> None:
    model = LoggingModel()

    model.log_metrics(
        {
            "loss": torch.tensor(3.0),
            "accuracy": 0.5,
            "train_lr": 0.001,
        },
        prefix="train",
    )

    assert model.current_metrics == {
        "train_loss": 3.0,
        "train_accuracy": 0.5,
        "train_lr": 0.001,
    }
    assert model.epoch_avg == {
        "train_loss": 3.0,
        "train_accuracy": 0.5,
        "train_lr": 0.001,
    }
