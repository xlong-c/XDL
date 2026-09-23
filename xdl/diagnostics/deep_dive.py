"""On-demand torch.profiler deep dive for suspected slow steps.

The recorder watches aggregate diagnostics signals and, once a trigger has
persisted for ``consecutive_required`` evaluations, captures a short
``torch.profiler`` window. The chrome trace is exported as
``deep_dive_<step>.json.gz`` under ``out_dir`` and a compact summary is derived
from ``key_averages()`` (top CUDA kernels plus explicit sync-source counts).

Everything degrades gracefully: if ``torch.profiler`` is missing or CUDA is
unavailable the recorder reports ``active() == False`` and never raises.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .records import GpuSummary, MemorySummary, PhaseAggregate, SyncSummary

logger = logging.getLogger(__name__)

# Number of consecutive trigger evaluations required before capturing.
_CONSECUTIVE_REQUIRED: int = 2
# Trigger thresholds.
_DATA_WAIT_FRAC: float = 0.25
_LOW_UTIL: float = 50.0
_VERY_LOW_UTIL: float = 30.0
_TAIL_RATIO: float = 2.5
_SAWTOOTH_MIN: int = 3
_TOP_KERNELS: int = 10


def _profiler_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return hasattr(torch, "profiler") and hasattr(torch.profiler, "profile")


class DeepDiveRecorder:
    """Triggers and records a single profiler capture on anomalous steps."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        wait: int = 0,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        max_captures: int = 1,
        enabled: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.wait = int(wait)
        self.warmup = int(warmup)
        self._active = int(active)
        self.repeat = int(repeat)
        self.max_captures = int(max_captures)
        self.enabled = bool(enabled)
        self._consecutive = 0
        self._captures = 0
        self._profiler: Optional[Any] = None
        self._capture_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_trigger(
        self,
        agg: PhaseAggregate,
        gpu: GpuSummary,
        syncs: SyncSummary,
        memory: MemorySummary,
    ) -> bool:
        """Return True once trigger conditions persist for two evaluations."""
        del syncs
        if not self.enabled or not self._can_capture():
            return False

        trigger = False
        if agg.frac("data_wait") > _DATA_WAIT_FRAC and (
            gpu.util_mean is None or gpu.util_mean < _LOW_UTIL
        ):
            trigger = True
        if (
            agg.p50.get("total", 0.0) > 0.0
            and agg.p95.get("total", 0.0) / agg.p50["total"] > _TAIL_RATIO
        ):
            trigger = True
        if (
            gpu.available
            and gpu.util_mean is not None
            and gpu.util_mean < _VERY_LOW_UTIL
        ):
            trigger = True
        if memory.sawtooth_count >= _SAWTOOTH_MIN:
            trigger = True

        if trigger:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= _CONSECUTIVE_REQUIRED

    def start(self, step: int) -> None:
        """Enter the profiler context for ``step``."""
        if not self.enabled or not self._can_capture():
            return
        if self._profiler is not None:
            return
        try:
            import torch
        except ImportError:  # pragma: no cover - guarded by _can_capture
            return
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if getattr(torch.cuda, "is_available", lambda: False)():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            schedule = torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self._active,
                repeat=self.repeat,
            )
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            )
            profiler.__enter__()
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.warning("Deep dive profiler failed to start: %s", exc)
            self._profiler = None
            return
        self._profiler = profiler
        self._capture_step = int(step)

    def step(self) -> None:
        """Advance the profiler schedule by one step."""
        if self._profiler is None:
            return
        try:
            self._profiler.step()
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.warning("Deep dive profiler step failed: %s", exc)

    def stop(self) -> Optional[Dict[str, Any]]:
        """Exit the profiler, export the trace, and return the summary."""
        profiler = self._profiler
        self._profiler = None
        if profiler is None:
            return None
        try:
            profiler.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.warning("Deep dive profiler failed to stop: %s", exc)
            return None
        self._captures += 1
        try:
            return self._export(profiler, self._capture_step)
        except Exception as exc:
            logger.warning("Deep dive export failed: %s", exc)
            return None

    def active(self) -> bool:
        """True while a capture is in progress."""
        return self._profiler is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _can_capture(self) -> bool:
        if not _profiler_available():
            return False
        return self._captures < self.max_captures and self._profiler is None

    def _export(self, profiler: Any, step: int) -> Dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.out_dir / f"deep_dive_{step}.json.gz"
        with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
            profiler.export_chrome_trace(handle)

        summary: Dict[str, Any] = {
            "step": step,
            "trace_path": str(trace_path),
            "top_kernels": self._top_kernels(profiler),
            "item_ops": 0,
            "stream_syncs": 0,
            "memcpy_dtoh": 0,
            "empty_cache_events": 0,
        }
        for event in profiler.key_averages():
            key = str(getattr(event, "key", ""))
            count = int(getattr(event, "count", 0))
            if key == "aten::_local_scalar_dense":
                summary["item_ops"] += count
            elif "cudaStreamSynchronize" in key:
                summary["stream_syncs"] += count
            elif "cudaMemcpyAsync" in key and ("DtoH" in key or "DeviceToHost" in key):
                summary["memcpy_dtoh"] += count
            elif "cudaEmptyCache" in key or "empty_cache" in key:
                summary["empty_cache_events"] += count
        return summary

    @staticmethod
    def _top_kernels(profiler: Any) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for event in profiler.key_averages():
            device_time = float(
                getattr(event, "self_device_time_total", 0.0)
                or getattr(event, "self_cuda_time_total", 0.0)
                or 0.0
            )
            if device_time <= 0.0:
                continue
            ranked.append(
                {
                    "name": str(getattr(event, "key", "")),
                    "self_cuda_time_us": device_time,
                    "count": int(getattr(event, "count", 0)),
                }
            )
        ranked.sort(key=lambda item: item["self_cuda_time_us"], reverse=True)
        return ranked[:_TOP_KERNELS]


__all__ = ["DeepDiveRecorder"]
