from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from xdl.model.generate import (
    IdentityRepresentor,
    RepresentationScatteringField,
    TBSMGenerator,
)
from xdl.trainer import TBSMCoreModel, Trainer
from xdl.task.pretrain import TBSMCoreModel as TaskTBSMCoreModel
from xdl.utils.registry import MODEL_REGISTRY


class TinyBackbone(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, 4)
        self.time_projection = nn.Linear(1, 4)
        self.conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.class_embedding(labels.long())
        condition = condition + self.time_projection(t.reshape(-1, 1))
        return self.conv(inputs) + condition[:, :3].reshape(-1, 3, 1, 1)


def build_tbsm(
    *,
    rho: float = 0.0,
    gradient_accumulation_steps: int = 1,
) -> tuple[TBSMCoreModel, Trainer]:
    generator = TBSMGenerator(TinyBackbone())
    field = RepresentationScatteringField(
        representor=IdentityRepresentor(),
        lambda_weight=0.0,
        rho=rho,
        num_classes=3,
        tracker_config={
            "hidden_dim": 8,
            "time_dim": 4,
            "depth": 1,
            "num_heads": 2,
        },
    )
    model = TBSMCoreModel(
        generator=generator,
        representation_fields=[field],
        gen_lr=0.05,
        tracker_lr=0.05,
        ema_decay=0.5,
        t_sampling=[1.0],
    )
    trainer = Trainer(
        max_epochs=1,
        device="cpu",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return model, trainer


def build_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randn(4, 3, 4, 4),
            torch.tensor([0, 1, 2, 0]),
        ),
        (
            torch.randn(4, 3, 4, 4),
            torch.tensor([1, 2, 0, 1]),
        ),
    ]


def test_tbsm_core_model_runs_through_trainer_and_updates_ema() -> None:
    model, trainer = build_tbsm()
    before = model.generator.backbone.conv.weight.detach().clone()
    ema_before = model.ema_generator.backbone.conv.weight.detach().clone()

    trainer.fit(model, build_batches())

    assert not torch.equal(model.generator.backbone.conv.weight, before)
    assert not torch.equal(model.ema_generator.backbone.conv.weight, ema_before)
    assert "train_loss" in model.last_epoch_avg
    assert "train_generator" in model.last_epoch_avg
    samples = model.sample(
        torch.randn(2, 3, 4, 4),
        torch.tensor([0, 1]),
    )
    assert samples.shape == (2, 3, 4, 4)
    assert float(samples.min()) >= 0.0
    assert float(samples.max()) <= 1.0


def test_tbsm_task_module_is_canonical_and_trainer_export_is_compatible() -> None:
    assert TBSMCoreModel is TaskTBSMCoreModel
    assert TBSMCoreModel.__module__ == "xdl.task.pretrain.tbsm"


def test_tbsm_public_model_registration() -> None:
    assert MODEL_REGISTRY.get("TBSMGenerator") is TBSMGenerator
    assert MODEL_REGISTRY.get("IdentityRepresentor") is IdentityRepresentor
    assert (
        MODEL_REGISTRY.get("RepresentationScatteringField")
        is RepresentationScatteringField
    )


def test_tbsm_core_model_runs_through_accelerate_cpu() -> None:
    model, trainer = build_tbsm()
    trainer = Trainer(
        max_epochs=1,
        device="cpu",
        accelerate_config={},
    )

    trainer.fit(model, build_batches())

    assert "train_loss" in model.last_epoch_avg
    assert len(model.optimizers) == 1


def test_tbsm_core_model_uses_tracker_optimizer_and_gradient_accumulation() -> None:
    model, trainer = build_tbsm(rho=1.0, gradient_accumulation_steps=2)
    tracker_before = (
        model.representation_fields[0].tracker.out_proj.weight.detach().clone()
    )

    trainer.fit(model, build_batches())

    tracker_after = model.representation_fields[0].tracker.out_proj.weight.detach()
    assert not torch.equal(tracker_after, tracker_before)
    assert len(model.optimizers) == 2
    assert model.optimizer_step == 1


def test_tbsm_core_model_accepts_mapping_batches() -> None:
    model, trainer = build_tbsm()
    batch = {
        "images": torch.randn(4, 3, 4, 4),
        "labels": torch.tensor([0, 1, 2, 0]),
    }

    trainer.fit(model, [batch])

    assert "train_loss" in model.last_epoch_avg


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([-0.1], "t_sampling"),
        ({"bad": 1}, "t_sampling"),
    ],
)
def test_tbsm_core_model_rejects_invalid_time_sampling(
    value: Any,
    message: str,
) -> None:
    expected_exception = TypeError if isinstance(value, dict) else ValueError
    with pytest.raises(expected_exception, match=message):
        TBSMCoreModel(
            generator=TBSMGenerator(TinyBackbone()),
            representation_fields=[],
            t_sampling=value,
        )
