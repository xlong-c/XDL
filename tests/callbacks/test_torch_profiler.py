import sys
import types
from typing import Any, Dict, List, Optional

import pytest

from xdl.callbacks import TorchProfilerCallback


class FakeProfilerInstance:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.entered = False
        self.exited = False
        self.steps = 0

    def __enter__(self) -> "FakeProfilerInstance":
        self.entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.exited = True

    def step(self) -> None:
        self.steps += 1


class FakeProfilerNamespace:
    class ProfilerActivity:
        CPU = "cpu"
        CUDA = "cuda"

    def __init__(self) -> None:
        self.schedules: List[Dict[str, int]] = []
        self.handlers: List[Dict[str, Optional[str]]] = []
        self.instances: List[FakeProfilerInstance] = []

    def schedule(self, **kwargs: int) -> Dict[str, int]:
        self.schedules.append(kwargs)
        return kwargs

    def tensorboard_trace_handler(
        self, log_dir: str, worker_name: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        handler = {"log_dir": log_dir, "worker_name": worker_name}
        self.handlers.append(handler)
        return handler

    def profile(self, **kwargs: Any) -> FakeProfilerInstance:
        instance = FakeProfilerInstance(kwargs)
        self.instances.append(instance)
        return instance


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True


def test_torch_profiler_callback_steps_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    profiler = FakeProfilerNamespace()
    fake_torch = types.SimpleNamespace(profiler=profiler, cuda=FakeCuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    callback = TorchProfilerCallback(
        log_dir=str(tmp_path / "profiler"),
        wait=0,
        warmup=0,
        active=2,
        repeat=1,
        worker_name="worker-0",
    )

    callback.on_train_start(trainer=object(), core_module=object())
    callback.on_train_batch_end(
        trainer=object(),
        core_module=object(),
        outputs={},
        batch=object(),
        batch_idx=0,
    )
    callback.on_train_end(trainer=object(), core_module=object())

    assert profiler.schedules == [
        {"wait": 0, "warmup": 0, "active": 2, "repeat": 1, "skip_first": 0}
    ]
    assert profiler.handlers == [
        {"log_dir": str(tmp_path / "profiler"), "worker_name": "worker-0"}
    ]
    assert profiler.instances[0].entered is True
    assert profiler.instances[0].steps == 1
    assert profiler.instances[0].exited is True
    assert callback.state_dict()["completed_steps"] == 1


def test_torch_profiler_callback_skips_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    profiler = FakeProfilerNamespace()
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch = types.SimpleNamespace(profiler=profiler, cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    callback = TorchProfilerCallback(log_dir=str(tmp_path), use_cuda=True)
    callback.on_train_start(trainer=object(), core_module=object())
    callback.on_train_end(trainer=object(), core_module=object())

    assert profiler.instances[0].config["activities"] == ["cpu"]


def test_torch_profiler_callback_validates_schedule() -> None:
    with pytest.raises(ValueError, match="active must be >= 1"):
        TorchProfilerCallback(active=0)
