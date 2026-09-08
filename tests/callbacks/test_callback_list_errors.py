"""回调错误处理行为测试."""

from typing import Any

import pytest

from xdl.callbacks import Callback, ModelCheckpoint
from xdl.post_training import SaveTrainableStateCallback
from xdl.callbacks.callback_list import CallbackList
from xdl.errors import TrainingError


class RaisingCallback(Callback):
    def on_train_epoch_end(self, trainer, core_module) -> None:
        del trainer, core_module
        raise ValueError("boom")


class FastFailRaisingCallback(RaisingCallback):
    fast_fail = True


class FakeTrainer:
    pass


class TypeErrorBodyCallback(Callback):
    """回调体内抛 TypeError, 用于验证不会被当作参数不匹配重试."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def on_train_batch_end(
        self,
        trainer,
        core_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, core_module, outputs, batch, batch_idx, dataloader_idx
        self.calls += 1
        raise TypeError("body failure")


def test_callback_body_type_error_is_not_retried() -> None:
    callback = TypeErrorBodyCallback()
    callback_list = CallbackList([callback])

    callback_list.train_batch_end(
        FakeTrainer(), object(), outputs={}, batch=object(), batch_idx=0
    )

    assert callback.calls == 1
    assert callback_list._error_count == 1


def test_fast_fail_callback_raises_immediately() -> None:
    callback_list = CallbackList([FastFailRaisingCallback()])
    with pytest.raises(TrainingError, match="fast_fail"):
        callback_list.epoch_end(FakeTrainer(), object())


def test_isolated_callback_error_reported_at_epoch_end() -> None:
    callback_list = CallbackList([RaisingCallback()])
    # 默认 fast_fail=False: 错误被记录但不中断.
    callback_list.epoch_end(FakeTrainer(), object())
    assert callback_list._error_count == 1

    # epoch 末汇总后再 raise.
    with pytest.raises(TrainingError, match="1 个回调失败"):
        callback_list.raise_errors()


def test_report_epoch_errors_clears_pending_raise() -> None:
    callback_list = CallbackList([RaisingCallback()])
    callback_list.epoch_end(FakeTrainer(), object())
    callback_list.report_epoch_errors()
    # 已汇总, 再次 raise 不应抛.
    callback_list.raise_errors()


def test_per_callback_fast_fail_does_not_swallow_save_errors() -> None:
    # 保存类回调默认 fast_fail, 保证 checkpoint 静默丢失不再发生.
    assert ModelCheckpoint.fast_fail is True
    assert SaveTrainableStateCallback.fast_fail is True
