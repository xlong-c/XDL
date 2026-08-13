"""callback 生命周期和错误策略测试."""

from __future__ import annotations

from typing import Any

import pytest

from xdl_jax import Callback, CallbackError, CallbackList


class RecordingCallback(Callback):
    def __init__(self, name: str, priority: int, events: list[str]) -> None:
        self.name = name
        self.priority = priority
        self.events = events

    def on_fit_start(self, trainer: Any, state: Any) -> None:
        del trainer, state
        self.events.append(self.name)


class RaisingCallback(Callback):
    def on_fit_start(self, trainer: Any, state: Any) -> None:
        del trainer, state
        raise RuntimeError("callback failure")


def test_callbacks_are_sorted_and_isolated_by_default() -> None:
    events: list[str] = []
    callbacks = CallbackList(
        [
            RecordingCallback("late", 20, events),
            RecordingCallback("early", 1, events),
            RaisingCallback(),
        ]
    )

    callbacks.invoke("on_fit_start", object(), object())

    assert events == ["early", "late"]
    assert len(callbacks.errors) == 1
    assert isinstance(callbacks.errors[0], CallbackError)


def test_callback_fast_fail_is_explicit() -> None:
    callback = RaisingCallback()
    callback.fast_fail = True
    callbacks = CallbackList([callback])

    with pytest.raises(CallbackError, match="callback failure"):
        callbacks.invoke("on_fit_start", object(), object())
