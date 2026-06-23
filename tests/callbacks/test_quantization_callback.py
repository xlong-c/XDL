from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from xdl.callbacks import QATLifecycleCallback


class FakeQuantModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observer_disabled = 0
        self.fake_quant_disabled = 0
        self.bn_frozen = 0

    def disable_observer(self) -> None:
        self.observer_disabled += 1

    def disable_fake_quant(self) -> None:
        self.fake_quant_disabled += 1

    def freeze_bn_stats(self) -> None:
        self.bn_frozen += 1


class WrappedTask:
    def __init__(self, model: nn.Module) -> None:
        self.model = model


def test_qat_lifecycle_callback_toggles_methods_after_threshold() -> None:
    module = FakeQuantModule()
    task = WrappedTask(nn.Sequential(module))
    callback = QATLifecycleCallback(
        disable_observer_epoch=2,
        freeze_bn_epoch=3,
        disable_fake_quant_epoch=4,
    )

    callback.on_train_epoch_start(SimpleNamespace(current_epoch=1), task)

    assert module.observer_disabled == 0
    assert module.fake_quant_disabled == 0
    assert module.bn_frozen == 0

    callback.on_train_epoch_start(SimpleNamespace(current_epoch=4), task)

    assert module.observer_disabled == 1
    assert module.fake_quant_disabled == 1
    assert module.bn_frozen == 1
    assert callback._state["last_epoch"] == 4
    assert callback._state["qat_lifecycle"] == {
        "observer_disabled": 1,
        "fake_quant_disabled": 1,
        "bn_frozen": 1,
    }


def test_qat_lifecycle_callback_uses_core_module_when_model_attr_is_missing() -> None:
    module = FakeQuantModule()
    callback = QATLifecycleCallback(disable_observer_epoch=0)

    callback.on_train_epoch_start(SimpleNamespace(current_epoch=0), module)

    assert module.observer_disabled == 1
