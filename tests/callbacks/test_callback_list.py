from typing import Any, List

from xdl.callbacks import Callback
from xdl.callbacks.callback_list import CallbackList


class RecordingCallback(Callback):
    def __init__(self, name: str, events: List[str], priority: int = 999) -> None:
        super().__init__(priority=priority)
        self.name = name
        self.events = events

    def on_train_start(self, trainer: Any, core_module: Any) -> None:
        del trainer, core_module
        self.events.append(self.name)


def test_callback_list_uses_callback_priority_by_default() -> None:
    events: List[str] = []
    late = RecordingCallback("late", events, priority=200)
    early = RecordingCallback("early", events, priority=10)

    callbacks = CallbackList([late, early])
    callbacks.train_start(trainer=object(), core_module=object())

    assert events == ["early", "late"]


def test_callback_list_explicit_priority_overrides_callback_priority() -> None:
    events: List[str] = []
    first = RecordingCallback("first", events, priority=900)
    second = RecordingCallback("second", events, priority=1)

    callbacks = CallbackList()
    callbacks.add_callback(first, priority=5)
    callbacks.add_callback(second, priority=50)
    callbacks.train_start(trainer=object(), core_module=object())

    assert events == ["first", "second"]
