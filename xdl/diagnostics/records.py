"""Diagnostics data records.

Pure dataclasses shared across the diagnostics subsystem. This module has no
dependency on Trainer, callbacks, or torch at import time, so the analyzer and
report renderer stay unit-testable on CPU with synthetic samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Phase timing records
# ---------------------------------------------------------------------------


@dataclass
class PhaseSample:
    """One training step phase breakdown, in seconds."""

    step: int
    epoch: int
    data_wait: float = 0.0
    h2d: float = 0.0
    train_step: float = 0.0
    nan_monitor: float = 0.0
    callbacks: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "data_wait": self.data_wait,
            "h2d": self.h2d,
            "train_step": self.train_step,
            "nan_monitor": self.nan_monitor,
            "callbacks": self.callbacks,
            "total": self.total,
        }


@dataclass
class PhaseAggregate:
    """Aggregated phase timing over a window, with per-phase shares."""

    steps: int = 0
    means: Dict[str, float] = field(default_factory=dict)
    p50: Dict[str, float] = field(default_factory=dict)
    p95: Dict[str, float] = field(default_factory=dict)
    fracs: Dict[str, float] = field(default_factory=dict)

    def frac(self, name: str) -> float:
        return float(self.fracs.get(name, 0.0))


# ---------------------------------------------------------------------------
# GPU / memory records
# ---------------------------------------------------------------------------


@dataclass
class GpuSample:
    """A single GPU sampling point."""

    timestamp: float
    utilization: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    memory_used: Optional[int] = None
    memory_total: Optional[int] = None
    temperature: Optional[float] = None

    @property
    def memory_used_frac(self) -> Optional[float]:
        if self.memory_used is None or not self.memory_total:
            return None
        return float(self.memory_used) / float(self.memory_total)


@dataclass
class GpuSummary:
    """Aggregated GPU statistics."""

    samples: int = 0
    util_mean: Optional[float] = None
    util_p50: Optional[float] = None
    util_p95: Optional[float] = None
    power_mean: Optional[float] = None
    power_max: Optional[float] = None
    power_limit: Optional[float] = None
    mem_used_frac_mean: Optional[float] = None
    temp_max: Optional[float] = None
    available: bool = False


@dataclass
class MemorySummary:
    """Reserved-memory sawtooth summary used to detect empty_cache thrash."""

    sawtooth_count: int = 0
    reserved_peak: int = 0
    reserved_min_after_peak: int = 0
    sawtooth_events: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sync counters
# ---------------------------------------------------------------------------


@dataclass
class SyncCounters:
    """Explicit counters for framework-owned per-step sync sources.

    Only framework-owned hot paths are counted here (CoreModel.log tensor
    extraction and the NaN grad-norm D2H). Arbitrary user .item()/.cpu() calls
    are intentionally not intercepted; the profiler deep dive enumerates them.
    """

    log_item: int = 0
    grad_norm: int = 0

    def reset(self) -> None:
        self.log_item = 0
        self.grad_norm = 0


@dataclass
class SyncSummary:
    """Per-step normalized sync counts."""

    steps: int = 0
    log_item_total: int = 0
    grad_norm_total: int = 0
    empty_cache_events: int = 0
    log_item_per_step: float = 0.0
    grad_norm_syncs_per_step: float = 0.0


# ---------------------------------------------------------------------------
# Findings and report
# ---------------------------------------------------------------------------


@dataclass
class Suggestion:
    """A concrete, actionable improvement for a finding."""

    title: str
    detail: str = ""
    code_hint: str = ""
    expected_gain: str = ""
    effort: str = "medium"  # "low" | "medium" | "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "detail": self.detail,
            "code_hint": self.code_hint,
            "expected_gain": self.expected_gain,
            "effort": self.effort,
        }


@dataclass
class Finding:
    """A root-cause hypothesis derived from measured signals."""

    id: str
    severity: str  # "high" | "medium" | "low"
    confidence: float
    title: str
    signals: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    suggestion: Optional[Suggestion] = None
    attributable_fraction: float = 0.0
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "title": self.title,
            "signals": self.signals,
            "evidence": self.evidence,
            "suggestion": self.suggestion.to_dict() if self.suggestion else None,
            "attributable_fraction": round(self.attributable_fraction, 4),
            "score": round(self.score, 4),
        }


@dataclass
class DiagnosticReport:
    """Top-level diagnostics report container."""

    run_meta: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    phases: Dict[str, Any] = field(default_factory=dict)
    gpu: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    syncs: Dict[str, Any] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    deep_dive: Optional[Dict[str, Any]] = None
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run": self.run_meta,
            "summary": self.summary,
            "phases": self.phases,
            "gpu": self.gpu,
            "memory": self.memory,
            "syncs": self.syncs,
            "findings": [f.to_dict() for f in self.findings],
            "suggestions": self.suggestions,
            "deep_dive": self.deep_dive,
        }
