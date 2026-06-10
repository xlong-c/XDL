from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pytest
import torch

from xdl.callbacks import Callback, ModelCheckpoint
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


class LifecycleRecordingCallback(Callback):
    def __init__(self, events: List[str]) -> None:
        super().__init__()
        self.events = events

    def setup(self, trainer: Trainer, core_module: CoreModel, stage: str) -> None:
        del trainer, core_module
        self.events.append(f"setup:{stage}")

    def on_fit_start(self, trainer: Trainer, core_module: CoreModel) -> None:
        del trainer, core_module
        self.events.append("fit_start")

    def on_train_start(self, trainer: Trainer, core_module: CoreModel) -> None:
        del trainer, core_module
        self.events.append("train_start")

    def on_train_end(self, trainer: Trainer, core_module: CoreModel) -> None:
        del trainer, core_module
        self.events.append("train_end")

    def on_fit_end(self, trainer: Trainer, core_module: CoreModel) -> None:
        del trainer, core_module
        self.events.append("fit_end")

    def teardown(self, trainer: Trainer, core_module: CoreModel, stage: str) -> None:
        del trainer, core_module
        self.events.append(f"teardown:{stage}")


class ManualOptimizationModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(1, 1, bias=False)
        self.step_flags: List[bool] = []

    def on_train_start(self) -> None:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.net.parameters(), lr=0.1)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        del batch_idx
        loss = self.net(batch).sum()
        self.step_flags.append(
            self.manual_optimization_step(
                loss,
                model=self.net,
                max_grad_norm=1.0,
            )
        )

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        pass


class FakePipeline:
    def __init__(self) -> None:
        self.devices: List[torch.device] = []

    def to(self, device: torch.device) -> "FakePipeline":
        self.devices.append(torch.device(device))
        return self


class DeviceObjectModel(RecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = FakePipeline()
        self.after_device_setup_called = False

    def configure_device_objects(self) -> Dict[str, Any]:
        return {"pipeline": self.pipeline}

    def on_after_device_setup(self) -> None:
        self.after_device_setup_called = True


class StatefulCallback(Callback):
    @property
    def state_key(self) -> str:
        return "stateful_callback"

    def __init__(self) -> None:
        super().__init__()
        self.value = 0

    def on_train_end(self, trainer: Trainer, core_module: CoreModel) -> None:
        del trainer, core_module
        self.value = 7

    def state_dict(self) -> Dict[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.value = int(state_dict["value"])


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


def test_trainer_invokes_fit_lifecycle_callbacks_in_order() -> None:
    events: List[str] = []
    callback = LifecycleRecordingCallback(events)
    trainer = Trainer(max_epochs=1, device="cpu", callbacks=[callback])
    model = RecordingModel()

    trainer.fit(model=model, train_dataloader=[0])

    assert events == [
        "setup:fit",
        "fit_start",
        "train_start",
        "train_end",
        "fit_end",
        "teardown:fit",
    ]


def test_float_val_check_interval_preserves_total_train_steps() -> None:
    model = RecordingModel()
    trainer = Trainer(max_epochs=2, device="cpu")

    trainer.fit(
        model=model,
        train_dataloader=[0, 1],
        val_dataloader=[2],
        val_check_interval=0.1,
    )

    assert len(model.train_batches) == 4
    assert len(model.validation_batches) == 4
    assert model.total_train_steps == 4


def test_invalid_val_check_interval_is_rejected() -> None:
    model = RecordingModel()
    trainer = Trainer(max_epochs=1, device="cpu")

    with pytest.raises(ValueError):
        trainer.fit(model=model, train_dataloader=[0, 1], val_check_interval=0.0)


def test_manual_optimization_step_handles_accumulation_boundaries() -> None:
    model = ManualOptimizationModel()
    trainer = Trainer(
        max_epochs=1,
        device="cpu",
        gradient_accumulation_steps=2,
    )

    trainer.fit(
        model=model,
        train_dataloader=[
            torch.ones(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1),
        ],
    )

    assert model.step_flags == [False, True, False]


def test_configure_device_objects_moves_extra_objects_and_calls_hook() -> None:
    model = DeviceObjectModel()
    trainer = Trainer(max_epochs=1, device="meta")

    trainer.fit(model=model, train_dataloader=[build_nested_batch(object())])

    assert model.pipeline.devices == [torch.device("meta")]
    assert model.after_device_setup_called is True


def test_trainer_load_checkpoint_restores_callback_state(tmp_path) -> None:
    stateful = StatefulCallback()
    checkpoint = ModelCheckpoint(
        dirpath=str(tmp_path),
        monitor=None,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=None,
        verbose=False,
    )
    trainer = Trainer(
        max_epochs=1,
        device="cpu",
        callbacks=[stateful, checkpoint],
    )
    model = ManualOptimizationModel()

    trainer.fit(model=model, train_dataloader=[torch.ones(1, 1), torch.ones(1, 1)])

    assert checkpoint.last_model_path is not None

    restored_callback = StatefulCallback()
    restore_trainer = Trainer(device="cpu", callbacks=[restored_callback])
    restored_model = ManualOptimizationModel()
    restore_trainer.load_checkpoint(restored_model, checkpoint.last_model_path)

    assert restored_callback.value == 7
