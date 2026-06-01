from dataclasses import dataclass
from typing import Any, List, Tuple

import torch

from xdl.trainer import CoreModel, Trainer


@dataclass
class NestedPayload:
    values: Tuple[torch.Tensor, torch.Tensor]
    marker: Any


def build_nested_batch(marker: Any) -> List[Any]:
    return [
        torch.ones(2, dtype=torch.float32),
        {
            "nested": (
                torch.zeros(3, dtype=torch.float32),
                NestedPayload(
                    values=(
                        torch.full((1,), 2.0, dtype=torch.float32),
                        torch.full((1,), 3.0, dtype=torch.float32),
                    ),
                    marker=marker,
                ),
            ),
            "text": "keep",
        },
    ]


class RecordingModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.train_batches: List[Any] = []
        self.validation_batches: List[Any] = []
        self.test_batches: List[Any] = []

    def on_train_start(self) -> None:
        pass

    def configure_optimizers(self) -> None:
        return None

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        self.train_batches.append(batch)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        self.validation_batches.append(batch)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        self.test_batches.append(batch)


class AccumulationRecordingModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.records: List[Tuple[int, int, int, bool, bool, bool]] = []

    def on_train_start(self) -> None:
        pass

    def configure_optimizers(self) -> None:
        return None

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx
        self.records.append(
            (
                self.micro_step,
                self.micro_step_in_accumulation,
                self.optimizer_step,
                self.is_accumulation_start,
                self.is_accumulation_boundary,
                self.should_optimizer_step,
            )
        )


class LegacyAccumulationRecordingModel(AccumulationRecordingModel):
    def __init__(self, gradient_accumulation_steps: int) -> None:
        super().__init__()
        self.gradient_accumulation_steps = gradient_accumulation_steps


def assert_nested_batch_on_meta(batch: Any, marker: Any) -> None:
    assert isinstance(batch, list)
    assert batch[0].device.type == "meta"

    nested_mapping = batch[1]
    assert isinstance(nested_mapping, dict)
    assert nested_mapping["text"] == "keep"

    nested_tuple = nested_mapping["nested"]
    assert isinstance(nested_tuple, tuple)
    assert nested_tuple[0].device.type == "meta"

    nested_payload = nested_tuple[1]
    assert isinstance(nested_payload, NestedPayload)
    assert isinstance(nested_payload.values, tuple)
    assert nested_payload.values[0].device.type == "meta"
    assert nested_payload.values[1].device.type == "meta"
    assert nested_payload.marker is marker


def test_transfer_to_device_recursively_preserves_container_types() -> None:
    trainer = Trainer(device="meta")
    marker = object()

    moved = trainer._transfer_to_device(build_nested_batch(marker))

    assert_nested_batch_on_meta(moved, marker)


def test_trainer_loops_use_recursive_transfer_for_train_val_and_test() -> None:
    marker = object()
    train_batch = build_nested_batch(marker)
    val_batch = build_nested_batch(marker)
    test_batch = build_nested_batch(marker)

    trainer = Trainer(max_epochs=1, device="meta")
    model = RecordingModel()

    trainer.fit(
        model=model,
        train_dataloader=[train_batch],
        val_dataloader=[val_batch],
    )
    trainer.test(model=model, test_dataloader=[test_batch])

    assert len(model.train_batches) == 1
    assert len(model.validation_batches) == 1
    assert len(model.test_batches) == 1

    assert_nested_batch_on_meta(model.train_batches[0], marker)
    assert_nested_batch_on_meta(model.validation_batches[0], marker)
    assert_nested_batch_on_meta(model.test_batches[0], marker)


def test_accumulation_helpers_track_micro_step_boundaries() -> None:
    model = AccumulationRecordingModel()
    trainer = Trainer(
        max_epochs=1,
        device="cpu",
        gradient_accumulation_steps=3,
    )

    trainer.fit(model=model, train_dataloader=[0, 1, 2, 3, 4])

    assert model.records == [
        (1, 1, 0, True, False, False),
        (2, 2, 0, False, False, False),
        (3, 3, 1, False, True, True),
        (4, 1, 1, True, False, False),
        (5, 2, 1, False, False, False),
    ]
    assert model.total_train_steps == 5
    assert model.micro_step == 5
    assert model.optimizer_step == 1
    assert trainer.micro_step == 5
    assert trainer.optimizer_step == 1
    assert trainer.micro_step_in_accumulation == 2
    assert not trainer.is_accumulation_boundary


def test_accumulation_helpers_respect_legacy_model_attribute() -> None:
    model = LegacyAccumulationRecordingModel(gradient_accumulation_steps=2)
    trainer = Trainer(max_epochs=1, device="cpu")

    trainer.fit(model=model, train_dataloader=[0, 1, 2])

    assert model.records == [
        (1, 1, 0, True, False, False),
        (2, 2, 1, False, True, True),
        (3, 1, 1, True, False, False),
    ]
