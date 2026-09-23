"""Tests for the DiagnosticsCallback lifecycle adapter (CPU only)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn

from xdl.callbacks import DiagnosticsCallback


class FakeTrainer:
    """Minimal duck-typed trainer for diagnostics lifecycle tests."""

    def __init__(self, callbacks: Optional[List[Any]] = None) -> None:
        self.global_step = 0
        self.current_epoch = 0
        self.batch_size = 4
        self.num_workers = 2
        self.max_epochs = 1
        self.device = "cpu"
        self.use_amp = False
        self.callbacks = callbacks or []
        self._main_process = True

    def is_main_process(self) -> bool:
        return self._main_process


class FakeCoreModule(nn.Module):
    """Tiny model exposing the attributes the callback summarizes."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def log(self, value: Any) -> None:
        del value


class FakeSampler:
    """Sampler stub used to simulate unavailable NVML."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.started = False

    @staticmethod
    def available() -> bool:
        return False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        pass

    def summary(self) -> Any:
        from xdl.diagnostics.records import GpuSummary

        return GpuSummary(available=False)


def _run_steps(
    callback: DiagnosticsCallback,
    trainer: FakeTrainer,
    core_module: FakeCoreModule,
    n: int = 3,
) -> None:
    for step in range(n):
        trainer.global_step = step + 1
        callback.on_train_batch_end(
            trainer=trainer,
            core_module=core_module,
            outputs=None,
            batch=None,
            batch_idx=step,
        )


def test_install_sets_frozen_attributes(tmp_path: Path) -> None:
    callback = DiagnosticsCallback(report_dir=str(tmp_path / "diag"))
    trainer = FakeTrainer()
    core_module = FakeCoreModule()

    callback.on_train_start(trainer, core_module)

    assert trainer._diag_enabled is True
    assert trainer._phase_timer is not None
    assert trainer._diag_counters is not None
    assert core_module._diag_counters is trainer._diag_counters

    callback.on_train_end(trainer, core_module)


def test_report_files_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "xdl.diagnostics.gpu_sampler.GpuSampler",
        FakeSampler,
    )
    report_dir = tmp_path / "diag"
    callback = DiagnosticsCallback(report_dir=str(report_dir), deep_dive=False)
    trainer = FakeTrainer()
    core_module = FakeCoreModule()

    callback.on_train_start(trainer, core_module)
    _run_steps(callback, trainer, core_module, n=3)
    callback.on_train_end(trainer, core_module)

    report_path = callback.state_dict()["report_path"]
    assert report_path is not None
    assert Path(report_path).exists()
    assert (report_dir / "diagnostics_report.txt").exists()

    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "phases" in payload
    assert "gpu" in payload


def test_degrades_without_pynvml_or_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "xdl.diagnostics.gpu_sampler.GpuSampler",
        FakeSampler,
    )
    callback = DiagnosticsCallback(report_dir=str(tmp_path / "diag"), deep_dive=False)
    trainer = FakeTrainer()
    core_module = FakeCoreModule()

    callback.on_train_start(trainer, core_module)
    _run_steps(callback, trainer, core_module, n=2)
    callback.on_train_end(trainer, core_module)

    assert callback.state_dict()["report_path"] is not None


def test_analysis_exception_does_not_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "xdl.diagnostics.gpu_sampler.GpuSampler",
        FakeSampler,
    )

    def boom(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("forced analysis failure")

    monkeypatch.setattr("xdl.diagnostics.analyzer.DiagnosticAnalyzer.analyze", boom)

    callback = DiagnosticsCallback(report_dir=str(tmp_path / "diag"), deep_dive=False)
    trainer = FakeTrainer()
    core_module = FakeCoreModule()

    callback.on_train_start(trainer, core_module)
    _run_steps(callback, trainer, core_module, n=2)
    callback.on_train_end(trainer, core_module)

    assert callback.state_dict()["report_path"] is None


def test_skips_when_not_main_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "xdl.diagnostics.gpu_sampler.GpuSampler",
        FakeSampler,
    )
    callback = DiagnosticsCallback(report_dir=str(tmp_path / "diag"), deep_dive=False)
    trainer = FakeTrainer()
    trainer._main_process = False
    core_module = FakeCoreModule()

    callback.on_train_start(trainer, core_module)
    _run_steps(callback, trainer, core_module, n=2)
    callback.on_train_end(trainer, core_module)

    assert callback.state_dict()["report_path"] is None


def test_disabled_callback_does_not_install(tmp_path: Path) -> None:
    callback = DiagnosticsCallback(report_dir=str(tmp_path / "diag"), enabled=False)
    trainer = FakeTrainer()
    core_module = FakeCoreModule()

    callback.on_train_start(trainer, core_module)

    assert not hasattr(trainer, "_diag_enabled")
    assert not hasattr(trainer, "_phase_timer")


def test_reserved_memory_sawtooth_via_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "xdl.diagnostics.gpu_sampler.GpuSampler",
        FakeSampler,
    )
    reserved = {"value": 100 * 1024 * 1024}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def memory_reserved() -> int:
            return reserved["value"]

    monkeypatch.setattr(torch, "cuda", FakeCuda)

    callback = DiagnosticsCallback(report_dir=str(tmp_path / "diag"), deep_dive=False)
    trainer = FakeTrainer()
    core_module = FakeCoreModule()
    callback.on_train_start(trainer, core_module)

    tracker = callback._memory_tracker
    assert tracker is not None
    reserved["value"] = 200 * 1024 * 1024
    _run_steps(callback, trainer, core_module, n=1)
    reserved["value"] = 50 * 1024 * 1024
    _run_steps(callback, trainer, core_module, n=1)

    assert tracker.summary().sawtooth_count >= 1


def test_skips_deep_dive_when_profiler_callback_present(tmp_path: Path) -> None:
    class TorchProfilerCallback:
        pass

    callback = DiagnosticsCallback(report_dir=str(tmp_path / "diag"), deep_dive=True)
    trainer = FakeTrainer(callbacks=[TorchProfilerCallback()])
    core_module = FakeCoreModule()

    callback.on_train_start(trainer, core_module)

    assert callback._deep_dive is None

    callback.on_train_end(trainer, core_module)


def test_missing_torch_profiler_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "xdl.diagnostics.gpu_sampler.GpuSampler",
        FakeSampler,
    )
    fake_torch = types.SimpleNamespace(cuda=torch.cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    callback = DiagnosticsCallback(report_dir=str(tmp_path / "diag"), deep_dive=True)
    trainer = FakeTrainer()
    core_module = FakeCoreModule()

    callback.on_train_start(trainer, core_module)
    _run_steps(callback, trainer, core_module, n=2)
    callback.on_train_end(trainer, core_module)

    assert callback._deep_dive is None
