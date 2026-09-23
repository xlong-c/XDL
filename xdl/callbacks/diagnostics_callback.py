"""Diagnostics lifecycle adapter callback.

Thin bridge between the XDL training loop and the pure-logic diagnostics
package: installs the frozen per-step hooks (``PhaseTimer``, ``SyncCounters``,
reserved-memory tracker), runs the background GPU sampler and on-demand deep
dive, and writes a report at the end of training.

Reporting is always best-effort: every analysis/report step is wrapped so a
failure degrades to a logged warning and never interrupts training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import Callback

if TYPE_CHECKING:
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer

logger = logging.getLogger(__name__)


def _torch_profiler_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    profiler = getattr(torch, "profiler", None)
    return profiler is not None and hasattr(profiler, "profile")


def _build_model_summary(core_module: Any) -> Dict[str, Any]:
    """Summarize trainable/frozen parameters and architecture traits."""
    summary: Dict[str, Any] = {
        "frozen_param_frac": 0.0,
        "has_conv_or_linear": False,
        "compiled": False,
    }
    try:
        import torch.nn as nn

        named_params = list(core_module.named_parameters())
        total = sum(int(p.numel()) for _, p in named_params)
        frozen = sum(
            int(p.numel()) for _, p in named_params if not bool(p.requires_grad)
        )
        if total > 0:
            summary["frozen_param_frac"] = float(frozen) / float(total)

        linear_types = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )
        summary["has_conv_or_linear"] = any(
            isinstance(module, linear_types) for module in core_module.modules()
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Model summary failed: %s", exc)

    summary["compiled"] = bool(
        getattr(core_module, "compiled", False)
        or getattr(core_module, "_compiled_call_impl", None) is not None
    )
    return summary


def _build_config_snapshot(trainer: Any) -> Dict[str, Any]:
    """Collect a small, serializable snapshot of training configuration.

    Batch shape and dataloader settings live on the train DataLoader, not on
    the Trainer, so they are read from ``trainer._train_dataloader``. Missing
    values stay absent rather than defaulting, so the analyzer can tell the
    difference between "not measured" and "explicitly zero".
    """
    snapshot: Dict[str, Any] = {}
    for key in ("max_epochs", "global_step", "device"):
        value = getattr(trainer, key, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            snapshot[key] = value

    accumulation = getattr(trainer, "gradient_accumulation_steps", None) or getattr(
        trainer, "accumulation_steps", None
    )
    if accumulation is not None:
        snapshot["accumulation_steps"] = int(accumulation)

    snapshot["nan_monitor"] = bool(getattr(trainer, "nan_monitor", False))
    precision = getattr(trainer, "precision", None)
    if precision is not None:
        snapshot["precision"] = str(precision)
    snapshot["amp"] = bool(getattr(trainer, "use_amp", False))

    loader = getattr(trainer, "_train_dataloader", None)
    if loader is not None:
        batch_size = getattr(loader, "batch_size", None)
        if isinstance(batch_size, int):
            snapshot["batch_size"] = batch_size
        num_workers = getattr(loader, "num_workers", None)
        if isinstance(num_workers, int):
            snapshot["num_workers"] = num_workers
        snapshot["pin_memory"] = bool(getattr(loader, "pin_memory", False))
        snapshot["persistent_workers"] = bool(
            getattr(loader, "persistent_workers", False)
        )
        prefetch = getattr(loader, "prefetch_factor", None)
        if isinstance(prefetch, int):
            snapshot["prefetch_factor"] = prefetch
    return snapshot


class DiagnosticsCallback(Callback):
    """Installs diagnostics hooks and writes a performance report.

    Priority defaults to ``1000`` so the callback runs after all other
    callbacks, ensuring its reserved-memory sample sees the end-of-step state.
    """

    def __init__(
        self,
        report_dir: str = "logs/diagnostics",
        *,
        enabled: bool = True,
        sample_interval_s: float = 0.5,
        deep_dive: bool = True,
        max_deep_dives: int = 1,
        window: int = 200,
        analyzer_config: Optional[Any] = None,
        priority: int = 1000,
    ) -> None:
        super().__init__(priority=priority)
        self.report_dir = Path(report_dir)
        self.enabled = bool(enabled)
        self.sample_interval_s = float(sample_interval_s)
        self.deep_dive = bool(deep_dive)
        self.max_deep_dives = int(max_deep_dives)
        self.window = int(window)
        self.analyzer_config = analyzer_config

        self._phase_timer: Optional[Any] = None
        self._counters: Optional[Any] = None
        self._memory_tracker: Optional[Any] = None
        self._sampler: Optional[Any] = None
        self._deep_dive: Optional[Any] = None
        self._reported = False
        self._state.update(
            {
                "report_path": None,
                "report_text_path": None,
                "deep_dive": None,
            }
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: "Trainer", core_module: "CoreModel", stage: str) -> None:
        del trainer, core_module, stage

    def on_train_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        if not self.enabled:
            return
        try:
            self._install(trainer, core_module)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Diagnostics callback failed to start: %s", exc)

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del core_module, outputs, batch, dataloader_idx
        if not self.enabled or self._phase_timer is None:
            return
        try:
            self._observe(trainer, batch_idx)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Diagnostics sample failed: %s", exc)

    def on_train_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        self._finalize(trainer, core_module)

    def teardown(
        self, trainer: "Trainer", core_module: "CoreModel", stage: str
    ) -> None:
        del stage
        self._finalize(trainer, core_module)

    def on_exception(
        self, trainer: "Trainer", core_module: "CoreModel", exception: Exception
    ) -> None:
        del trainer, core_module, exception
        self._stop_deep_dive()
        self._stop_sampler()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _install(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        from xdl.diagnostics.gpu_sampler import GpuSampler
        from xdl.diagnostics.phase_timer import PhaseTimer
        from xdl.diagnostics.records import SyncCounters
        from xdl.diagnostics.sync_tracker import ReservedMemoryTracker

        timer = PhaseTimer(window=self.window)
        counters = SyncCounters()
        tracker = ReservedMemoryTracker(window=self.window)

        self._phase_timer = timer
        self._counters = counters
        self._memory_tracker = tracker

        trainer._diag_enabled = True
        trainer._phase_timer = timer
        trainer._diag_counters = counters
        core_module._diag_counters = counters

        self._sampler = GpuSampler(interval_s=self.sample_interval_s)
        self._sampler.start()

        if self.deep_dive and not self._has_profiler_callback(trainer):
            from xdl.diagnostics.deep_dive import DeepDiveRecorder

            if _torch_profiler_available():
                self._deep_dive = DeepDiveRecorder(
                    self.report_dir,
                    max_captures=self.max_deep_dives,
                )

    @staticmethod
    def _has_profiler_callback(trainer: "Trainer") -> bool:
        callbacks = getattr(trainer, "callbacks", None)
        if callbacks is None:
            return False
        iterable = getattr(callbacks, "callbacks", callbacks)
        try:
            for callback in iterable:
                if callback.__class__.__name__ == "TorchProfilerCallback":
                    return True
        except TypeError:
            return False
        return False

    def _observe(self, trainer: "Trainer", batch_idx: int) -> None:
        del batch_idx
        step = int(getattr(trainer, "global_step", 0) or 0)

        self._sample_reserved_memory(step)

        if self._deep_dive is not None and self._deep_dive.active():
            self._deep_dive.step()

        if step % max(1, self.window) == 0:
            self._maybe_start_deep_dive(step)

    def _sample_reserved_memory(self, step: int) -> None:
        tracker = self._memory_tracker
        if tracker is None:
            return
        try:
            import torch

            if not torch.cuda.is_available():
                return
            tracker.observe(int(torch.cuda.memory_reserved()), step=step)
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.debug("Reserved memory sample failed: %s", exc)

    def _maybe_start_deep_dive(self, step: int) -> None:
        recorder = self._deep_dive
        if recorder is None or recorder.active():
            return
        timer = self._phase_timer
        agg = timer.snapshot() if timer is not None else _phase_unavailable()
        sampler = self._sampler
        gpu = sampler.summary() if sampler is not None else _gpu_unavailable()
        syncs = self._sync_summary()
        tracker = self._memory_tracker
        memory = tracker.summary() if tracker is not None else _memory_unavailable()
        if recorder.should_trigger(agg, gpu, syncs, memory):
            recorder.start(step)

    def _sync_summary(self) -> Any:
        from xdl.diagnostics.records import SyncSummary

        counters = self._counters
        steps = len(self._phase_timer) if self._phase_timer is not None else 0
        if counters is None:
            return SyncSummary(steps=steps)
        log_item = int(counters.log_item)
        grad_norm = int(counters.grad_norm)
        denom = steps if steps > 0 else 1
        return SyncSummary(
            steps=steps,
            log_item_total=log_item,
            grad_norm_total=grad_norm,
            log_item_per_step=log_item / denom,
            grad_norm_syncs_per_step=grad_norm / denom,
        )

    def _stop_sampler(self) -> None:
        if self._sampler is not None:
            try:
                self._sampler.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("GPU sampler stop failed: %s", exc)

    def _stop_deep_dive(self) -> Optional[Dict[str, Any]]:
        if self._deep_dive is None:
            return None
        summary: Optional[Dict[str, Any]] = None
        try:
            summary = self._deep_dive.stop()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Deep dive stop failed: %s", exc)
        return summary

    def _finalize(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        if self._reported:
            return
        self._reported = True
        deep_dive_summary = self._stop_deep_dive()
        self._stop_sampler()
        self._state["deep_dive"] = deep_dive_summary

        if not self._is_main_process(trainer):
            return
        try:
            self._report(trainer, core_module, deep_dive_summary)
        except Exception as exc:
            logger.warning("Diagnostics report generation failed: %s", exc)

    @staticmethod
    def _is_main_process(trainer: "Trainer") -> bool:
        checker = getattr(trainer, "is_main_process", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:  # pragma: no cover - defensive
                return True
        return True

    def _report(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        deep_dive_summary: Optional[Dict[str, Any]],
    ) -> None:
        from xdl.diagnostics.analyzer import (
            AnalysisContext,
            AnalyzerConfig,
            DiagnosticAnalyzer,
        )
        from xdl.diagnostics.report import build_report, write_report

        timer = self._phase_timer
        agg = timer.snapshot() if timer is not None else None
        if agg is None:
            from xdl.diagnostics.records import PhaseAggregate

            agg = PhaseAggregate()
        sampler = self._sampler
        gpu = sampler.summary() if sampler is not None else _gpu_unavailable()
        syncs = self._sync_summary()
        tracker = self._memory_tracker
        memory = tracker.summary() if tracker is not None else _memory_unavailable()
        config = _build_config_snapshot(trainer)
        model = _build_model_summary(core_module)

        analyzer_config = self.analyzer_config or AnalyzerConfig()
        analyzer = DiagnosticAnalyzer(config=analyzer_config)
        ctx = AnalysisContext(
            phases=agg,
            gpu=gpu,
            syncs=syncs,
            memory=memory,
            config=config,
            model=model,
            deep_dive=deep_dive_summary,
        )
        findings = analyzer.analyze(ctx)
        run_meta = {
            "global_step": int(getattr(trainer, "global_step", 0) or 0),
            "epoch": int(getattr(trainer, "current_epoch", 0) or 0),
            "report_dir": str(self.report_dir),
        }
        report = build_report(
            run_meta=run_meta,
            phases=agg,
            gpu=gpu,
            syncs=syncs,
            memory=memory,
            findings=findings,
            deep_dive=deep_dive_summary,
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)
        json_path, text_path = write_report(str(self.report_dir), report)
        self._state["report_path"] = json_path
        self._state["report_text_path"] = text_path
        logger.info("Diagnostics report written to %s", json_path)


def _gpu_unavailable() -> Any:
    from xdl.diagnostics.records import GpuSummary

    return GpuSummary(available=False)


def _memory_unavailable() -> Any:
    from xdl.diagnostics.records import MemorySummary

    return MemorySummary()


def _phase_unavailable() -> Any:
    from xdl.diagnostics.records import PhaseAggregate

    return PhaseAggregate()


__all__: List[str] = ["DiagnosticsCallback"]
