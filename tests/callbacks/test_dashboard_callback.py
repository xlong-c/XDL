"""DashboardCallback event payload tests."""

from __future__ import annotations

from typing import Any, Dict

from xdl.callbacks import DashboardCallback


class FakeTrainer:
    current_epoch = 3
    global_step = 7
    should_stop = False
    control = None


class FakeModel:
    current_metrics = {"train_loss": 0.25, "lr": 1e-3, "ignored": "text"}
    last_epoch_avg = {"train_loss": 0.4, "val_accuracy": 0.9}


def test_callback_emits_scalar_metrics_and_reliable_lifecycle_events() -> None:
    events: list[tuple[str, Dict[str, Any], bool]] = []
    callback = DashboardCallback(
        lambda event_type, payload, reliable: events.append(
            (event_type, payload, reliable)
        ),
        every_n_steps=2,
        interval_s=0.0,
    )
    trainer = FakeTrainer()
    model = FakeModel()

    callback.on_fit_start(trainer, model)
    callback.on_train_start(trainer, model)
    callback.on_train_batch_end(trainer, model, {}, None, 0)
    trainer.global_step = 8
    callback.on_train_batch_end(trainer, model, {}, None, 1)
    callback.on_train_epoch_end(trainer, model)

    assert [event[0] for event in events] == [
        "run_state",
        "run_state",
        "metrics",
        "train_epoch_end",
    ]
    metric_event = events[2]
    assert metric_event[1]["global_step"] == 8
    assert metric_event[1]["metrics"] == {"train_loss": 0.25, "lr": 1e-3}
    assert metric_event[2] is False
    assert events[0][2] is True


def test_callback_disables_broken_sink_without_raising() -> None:
    attempts = 0

    def broken_sink(event_type: str, payload: Dict[str, Any], reliable: bool) -> None:
        nonlocal attempts
        del event_type, payload, reliable
        attempts += 1
        raise OSError("event queue is closed")

    callback = DashboardCallback(broken_sink)
    callback.on_fit_start(FakeTrainer(), FakeModel())
    callback.on_train_start(FakeTrainer(), FakeModel())

    assert attempts == 1
