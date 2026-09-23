"""Per-step phase timer for the training loop.

The trainer writes lightweight timestamp marks into a PhaseTimer; each step is
finalized into a PhaseSample. Cost per mark is one bool check plus one
``time.perf_counter()`` call, well below the always-on overhead budget.
"""

from __future__ import annotations

import time
from collections import deque
from statistics import mean
from typing import Callable, Deque, Dict, List, Optional

from .records import PhaseAggregate, PhaseSample

# Ordered mark names for a single training step.
MARK_SEQUENCE: tuple = (
    "step_begin",
    "data_ready",
    "h2d_done",
    "step_done",
    "nan_done",
    "batch_end",
)

# Phase name -> (start mark, end mark).
PHASE_MARKS: Dict[str, tuple] = {
    "data_wait": ("step_begin", "data_ready"),
    "h2d": ("data_ready", "h2d_done"),
    "train_step": ("h2d_done", "step_done"),
    "nan_monitor": ("step_done", "nan_done"),
    "callbacks": ("nan_done", "batch_end"),
}


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolated percentile; returns 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


class PhaseTimer:
    """Collects phase timestamps and aggregates them over a rolling window."""

    def __init__(
        self,
        clock: Callable[[], float] = time.perf_counter,
        window: int = 200,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._clock = clock
        self._window = window
        self._samples: Deque[PhaseSample] = deque(maxlen=window)
        self._marks: Dict[str, float] = {}
        self._step: int = 0
        self._epoch: int = 0

    def mark(self, name: str) -> None:
        """Record a timestamp for ``name``."""
        self._marks[name] = self._clock()

    def set_step(self, step: int, epoch: int) -> None:
        self._step = step
        self._epoch = epoch

    def finish_step(
        self, step: Optional[int] = None, epoch: Optional[int] = None
    ) -> Optional[PhaseSample]:
        """Finalize the current step from collected marks.

        Returns None when the minimum required marks are absent.
        """
        if "step_begin" not in self._marks or "batch_end" not in self._marks:
            self._marks.clear()
            return None

        resolved_step = self._step if step is None else step
        resolved_epoch = self._epoch if epoch is None else epoch

        sample = PhaseSample(
            step=resolved_step,
            epoch=resolved_epoch,
            total=max(0.0, self._marks["batch_end"] - self._marks["step_begin"]),
        )
        for phase, (start_mark, end_mark) in PHASE_MARKS.items():
            if start_mark in self._marks and end_mark in self._marks:
                value = self._marks[end_mark] - self._marks[start_mark]
            else:
                value = 0.0
            setattr(sample, phase, max(0.0, value))

        self._samples.append(sample)
        self._marks.clear()
        return sample

    def snapshot(self) -> PhaseAggregate:
        """Aggregate the rolling window into means, percentiles, and shares."""
        samples = list(self._samples)
        agg = PhaseAggregate(steps=len(samples))
        if not samples:
            return agg

        phase_names = [
            "data_wait",
            "h2d",
            "train_step",
            "nan_monitor",
            "callbacks",
            "total",
        ]
        for name in phase_names:
            values = [float(getattr(s, name)) for s in samples]
            agg.means[name] = mean(values)
            agg.p50[name] = _percentile(values, 50.0)
            agg.p95[name] = _percentile(values, 95.0)

        total_mean = agg.means.get("total", 0.0)
        if total_mean > 0.0:
            for name in phase_names:
                if name == "total":
                    continue
                agg.fracs[name] = agg.means[name] / total_mean
        return agg

    def reset(self) -> None:
        self._samples.clear()
        self._marks.clear()

    def __len__(self) -> int:
        return len(self._samples)
