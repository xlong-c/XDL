"""XDL training performance diagnostics.

Built-in GPU-utilization diagnostics for XDL training. The subsystem is split
into a pure-logic package (this package) and a thin lifecycle adapter
(``xdl.callbacks.diagnostics_callback.DiagnosticsCallback``).

Public surface:
- ``PhaseTimer`` / ``PhaseSample`` / ``PhaseAggregate``: per-step phase timing.
- ``GpuSampler`` / ``GpuSample`` / ``GpuSummary``: background NVML sampling.
- ``ReservedMemoryTracker`` / ``MemorySummary``: empty_cache thrash detection.
- ``SyncCounters`` / ``SyncSummary``: framework-owned per-step sync counters.
- ``DiagnosticAnalyzer``: rule engine mapping signals to root-cause findings.
- ``DeepDiveRecorder``: on-demand torch.profiler capture.
- ``build_report`` / ``render_text`` / ``write_report``: report rendering.
"""

from __future__ import annotations

from .analyzer import AnalysisContext, AnalyzerConfig, DiagnosticAnalyzer
from .deep_dive import DeepDiveRecorder
from .gpu_sampler import GpuSampler
from .phase_timer import PhaseTimer
from .records import (
    DiagnosticReport,
    Finding,
    GpuSample,
    GpuSummary,
    MemorySummary,
    PhaseAggregate,
    PhaseSample,
    Suggestion,
    SyncCounters,
    SyncSummary,
)
from .report import build_report, render_text, write_report
from .sync_tracker import ReservedMemoryTracker

__all__ = [
    "PhaseTimer",
    "PhaseSample",
    "PhaseAggregate",
    "GpuSample",
    "GpuSummary",
    "MemorySummary",
    "SyncCounters",
    "SyncSummary",
    "Suggestion",
    "Finding",
    "DiagnosticReport",
    "ReservedMemoryTracker",
    "GpuSampler",
    "DeepDiveRecorder",
    "AnalyzerConfig",
    "AnalysisContext",
    "DiagnosticAnalyzer",
    "build_report",
    "render_text",
    "write_report",
]
