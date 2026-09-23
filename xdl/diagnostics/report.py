"""Diagnostics report assembly and rendering.

Turns the measured records plus the ranked findings into a
:class:`~xdl.diagnostics.records.DiagnosticReport`, renders a human-readable
text summary, and writes both the JSON and text artifacts to disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from .records import (
    DiagnosticReport,
    Finding,
    GpuSummary,
    MemorySummary,
    PhaseAggregate,
    SyncSummary,
)

JSON_FILENAME = "diagnostics_report.json"
TEXT_FILENAME = "diagnostics_report.txt"

_SEVERITY_ORDER: Dict[str, int] = {"high": 0, "medium": 1, "low": 2}


def _asdict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    return asdict(value)


def _build_summary(
    run_meta: Dict[str, Any],
    phases: PhaseAggregate,
    gpu: GpuSummary,
) -> Dict[str, Any]:
    return {
        "steps": phases.steps,
        "wall_s": run_meta.get("wall_s"),
        "device": run_meta.get("device"),
        "gpu_util_mean": gpu.util_mean,
        "power_mean": gpu.power_mean,
        "gpu_available": gpu.available,
        "finding_count": 0,
        "top_finding": None,
    }


def _rank_suggestions(findings: List[Finding]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for rank, finding in enumerate(findings, start=1):
        suggestion = finding.suggestion
        ranked.append(
            {
                "rank": rank,
                "id": finding.id,
                "title": suggestion.title if suggestion else finding.title,
                "expected_gain": suggestion.expected_gain if suggestion else "",
                "effort": suggestion.effort if suggestion else "medium",
            }
        )
    return ranked


def build_report(
    *,
    run_meta: Dict[str, Any],
    phases: PhaseAggregate,
    gpu: GpuSummary,
    syncs: SyncSummary,
    memory: MemorySummary,
    findings: List[Finding],
    deep_dive: Optional[Dict[str, Any]] = None,
) -> DiagnosticReport:
    """Assemble a :class:`DiagnosticReport` from measured records and findings."""
    summary = _build_summary(run_meta, phases, gpu)
    summary["finding_count"] = len(findings)
    summary["top_finding"] = findings[0].id if findings else None
    return DiagnosticReport(
        run_meta=dict(run_meta),
        summary=summary,
        phases=_asdict(phases),
        gpu=_asdict(gpu),
        memory=_asdict(memory),
        syncs=_asdict(syncs),
        findings=list(findings),
        suggestions=_rank_suggestions(findings),
        deep_dive=deep_dive,
        schema_version=1,
    )


def _fmt_optional(value: Optional[float], suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _render_header(report: DiagnosticReport) -> List[str]:
    summary = report.summary or {}
    run = report.run_meta or {}
    device = summary.get("device", run.get("device", "n/a"))
    wall_s = summary.get("wall_s", run.get("wall_s"))
    lines = [
        "=" * 64,
        "XDL Training Diagnostics Report",
        "=" * 64,
        f"steps:          {summary.get('steps', 0)}",
        f"wall_s:         {_fmt_optional(wall_s)}",
        f"device:         {device}",
        f"GPU util mean:  {_fmt_optional(summary.get('gpu_util_mean'), '%', 1)}",
        f"power mean:     {_fmt_optional(summary.get('power_mean'), ' W', 1)}",
        "",
    ]
    return lines


def _render_phase_table(report: DiagnosticReport) -> List[str]:
    phases = report.phases or {}
    fracs = phases.get("fracs", {}) or {}
    means = phases.get("means", {}) or {}
    p50 = phases.get("p50", {}) or {}
    p95 = phases.get("p95", {}) or {}
    names = ["data_wait", "h2d", "train_step", "nan_monitor", "callbacks", "total"]
    lines = ["Phase Timing", "-" * 64]
    lines.append(f"{'phase':<14}{'mean_s':>10}{'p50_s':>10}{'p95_s':>10}{'share':>10}")
    for name in names:
        if name not in means and name not in fracs:
            continue
        share = fracs.get(name)
        share_txt = f"{share:.1%}" if share is not None else "-"
        lines.append(
            f"{name:<14}{means.get(name, 0.0):>10.4f}"
            f"{p50.get(name, 0.0):>10.4f}{p95.get(name, 0.0):>10.4f}"
            f"{share_txt:>10}"
        )
    lines.append("")
    return lines


def _render_gpu_table(report: DiagnosticReport) -> List[str]:
    gpu = report.gpu or {}
    memory = report.memory or {}
    syncs = report.syncs or {}
    lines = ["GPU / Memory / Syncs", "-" * 64]
    lines.append(f"gpu available:      {gpu.get('available', False)}")
    lines.append(f"samples:            {gpu.get('samples', 0)}")
    lines.append(
        f"util p50 / p95:     {_fmt_optional(gpu.get('util_p50'), '%', 1)}"
        f" / {_fmt_optional(gpu.get('util_p95'), '%', 1)}"
    )
    lines.append(
        f"power max / limit:  {_fmt_optional(gpu.get('power_max'), ' W', 1)}"
        f" / {_fmt_optional(gpu.get('power_limit'), ' W', 1)}"
    )
    lines.append(
        f"mem used frac mean: {_fmt_optional(gpu.get('mem_used_frac_mean'), '', 3)}"
    )
    lines.append(f"temp max:           {_fmt_optional(gpu.get('temp_max'), ' C', 1)}")
    lines.append(f"sawtooth count:     {memory.get('sawtooth_count', 0)}")
    lines.append(f"log_item/step:      {syncs.get('log_item_per_step', 0.0):.2f}")
    lines.append(
        f"grad_norm syncs/step:{syncs.get('grad_norm_syncs_per_step', 0.0):>6.2f}"
    )
    lines.append("")
    return lines


def _render_findings(report: DiagnosticReport) -> List[str]:
    lines = ["Findings", "-" * 64]
    if not report.findings:
        lines.append("No issues detected")
        lines.append("")
        return lines
    for rank, finding in enumerate(report.findings, start=1):
        lines.append(
            f"[{rank}] {finding.id}  severity={finding.severity}  "
            f"confidence={finding.confidence:.2f}  score={finding.score:.2f}"
        )
        lines.append(f"    {finding.title}")
        if finding.signals:
            signals = ", ".join(
                f"{key}={value}" for key, value in finding.signals.items()
            )
            lines.append(f"    signals: {signals}")
        for item in finding.evidence:
            lines.append(f"    - {item}")
        suggestion = finding.suggestion
        if suggestion is not None:
            lines.append(
                f"    suggestion: {suggestion.title} "
                f"(effort={suggestion.effort}, gain={suggestion.expected_gain})"
            )
            if suggestion.detail:
                lines.append(f"      {suggestion.detail}")
            if suggestion.code_hint:
                lines.append(f"      code: {suggestion.code_hint}")
        lines.append("")
    return lines


def render_text(report: DiagnosticReport) -> str:
    """Render a readable multi-section text report."""
    lines: List[str] = []
    lines.extend(_render_header(report))
    lines.extend(_render_phase_table(report))
    lines.extend(_render_gpu_table(report))
    lines.extend(_render_findings(report))
    return "\n".join(lines).rstrip() + "\n"


def write_report(report_dir: str, report: DiagnosticReport) -> Tuple[str, str]:
    """Write JSON and text artifacts into ``report_dir``.

    Returns ``(json_path, txt_path)``.
    """
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, JSON_FILENAME)
    txt_path = os.path.join(report_dir, TEXT_FILENAME)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(render_text(report))
    return json_path, txt_path
