"""CallbackList.train_batch_start 的 batch 替换契约测试."""

from typing import Any

from xdl.callbacks import Callback
from xdl.callbacks.callback_list import CallbackList


class NoReplaceCallback(Callback):
    """不替换 batch 的回调 (返回 None)."""

    def on_train_batch_start(
        self, trainer: Any, core_module: Any, batch: Any, batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return None


class PassthroughCallback(Callback):
    """原样返回 batch (非 None, 等价于不替换)."""

    def on_train_batch_start(
        self, trainer: Any, core_module: Any, batch: Any, batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return batch


class ReplaceBatchCallback(Callback):
    """返回注入的替换 batch (模拟 RolloutCallback 注入 RolloutBatch)."""

    def __init__(self, replacement: Any) -> None:
        super().__init__()
        self.replacement = replacement

    def on_train_batch_start(
        self, trainer: Any, core_module: Any, batch: Any, batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return self.replacement


class ObserveBatchCallback(Callback):
    """记录它收到的 batch, 验证回调按链路传递替换结果."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: Any = None

    def on_train_batch_start(
        self, trainer: Any, core_module: Any, batch: Any, batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.seen = batch
        return None


def test_train_batch_start_replaces_batch() -> None:
    replacement = {"injected": True}
    callbacks = CallbackList([NoReplaceCallback(), ReplaceBatchCallback(replacement)])
    result = callbacks.train_batch_start(
        trainer=None, core_module=None, batch=[1, 2], batch_idx=0
    )
    assert result is replacement


def test_train_batch_start_without_replacement_returns_original() -> None:
    callbacks = CallbackList([NoReplaceCallback(), PassthroughCallback()])
    original = [1, 2]
    result = callbacks.train_batch_start(
        trainer=None, core_module=None, batch=original, batch_idx=0
    )
    assert result is original


def test_train_batch_start_last_replacement_wins() -> None:
    first = {"stage": 1}
    second = {"stage": 2}
    callbacks = CallbackList(
        [ReplaceBatchCallback(first), ReplaceBatchCallback(second)]
    )
    result = callbacks.train_batch_start(
        trainer=None, core_module=None, batch=[0], batch_idx=0
    )
    assert result is second


def test_train_batch_start_passes_replacement_to_later_callbacks() -> None:
    replacement = {"stage": 1}
    observer = ObserveBatchCallback()
    callbacks = CallbackList([ReplaceBatchCallback(replacement), observer])

    result = callbacks.train_batch_start(
        trainer=None, core_module=None, batch=[0], batch_idx=0
    )

    assert result is replacement
    assert observer.seen is replacement
