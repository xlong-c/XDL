"""Tests for diagnostics report assembly, rendering, and writing.

All records are synthetic; the tests run on CPU with no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from xdl.diagnostics.records import (
    DiagnosticReport,
    Finding,
    GpuSummary,
    MemorySummary,
    PhaseAggregate,
    Suggestion,
    SyncSummary,
)
from xdl.diagnostics.report import build_report, render_text, write_report


def make_phases() -> PhaseAggregate:
    return PhaseAggregate(
        steps=50,
        means={
            "data_wait": 0.3,
            "h2d": 0.05,
            "train_step": 0.6,
            "nan_monitor": 0.02,
            "callbacks": 0.03,
            "total": 1.0,
        },
        p50={"data_wait": 0.29, "total": 1.0},
        p95={"data_wait": 0.45, "total": 1.2},
        fracs={
            "data_wait": 0.30,
            "h2d": 0.05,
            "train_step": 0.60,
            "nan_monitor": 0.02,
            "callbacks": 0.03,
        },
    )


def make_gpu() -> GpuSummary:
    return GpuSummary(
        samples=100,
        util_mean=42.5,
        util_p50=40.0,
        util_p95=70.0,
        power_mean=150.0,
        power_max=180.0,
        power_limit=250.0,
        mem_used_frac_mean=0.55,
        temp_max=72.0,
        available=True,
    )


def make_syncs() -> SyncSummary:
    return SyncSummary(
        steps=50,
        log_item_total=200,
        grad_norm_total=50,
        empty_cache_events=2,
        log_item_per_step=4.0,
        grad_norm_syncs_per_step=1.0,
    )


def make_findings() -> List[Finding]:
    return [
        Finding(
            id="dataloader_starvation",
            severity="high",
            confidence=0.8,
            title="DataLoader starvation",
            signals={"data_wait_frac": 0.3},
            evidence=["data_wait is 30.0% of wall time"],
            suggestion=Suggestion(
                title="Feed the GPU faster",
                detail="More workers and pinned memory.",
                code_hint="num_workers=8; pin_memory=True",
                expected_gain="Recover wall time",
                effort="low",
            ),
            attributable_fraction=0.3,
            score=0.72,
        ),
        Finding(
            id="no_compile",
            severity="low",
            confidence=0.45,
            title="Model not compiled",
            signals={"train_step_frac": 0.6},
            evidence=["train_step is 60.0% of wall time"],
            suggestion=Suggestion(
                title="Try torch.compile",
                detail="Fuse pointwise ops.",
                code_hint="model = torch.compile(model)",
                expected_gain="Modest speedup",
                effort="medium",
            ),
            attributable_fraction=0.6,
            score=0.81,
        ),
    ]


def build_sample_report(findings: List[Finding] | None = None) -> DiagnosticReport:
    return build_report(
        run_meta={
            "wall_s": 123.4,
            "device": "cuda:0",
            "run_name": "unit-test",
        },
        phases=make_phases(),
        gpu=make_gpu(),
        syncs=make_syncs(),
        memory=MemorySummary(
            sawtooth_count=4,
            reserved_peak=8 * 1024**3,
            reserved_min_after_peak=2 * 1024**3,
            sawtooth_events=[10, 20, 30, 40],
        ),
        findings=make_findings() if findings is None else findings,
        deep_dive={"empty_cache_events": 2},
    )


class TestBuildReport:
    def test_populates_core_fields(self) -> None:
        report = build_sample_report()
        assert report.run_meta["device"] == "cuda:0"
        assert report.summary["steps"] == 50
        assert report.summary["finding_count"] == 2
        assert report.summary["top_finding"] == "dataloader_starvation"
        assert report.phases["fracs"]["data_wait"] == 0.30
        assert report.gpu["util_mean"] == 42.5
        assert report.syncs["log_item_per_step"] == 4.0
        assert report.memory["sawtooth_count"] == 4
        assert report.schema_version == 1
        assert report.deep_dive == {"empty_cache_events": 2}

    def test_suggestions_ranked_from_findings_order(self) -> None:
        report = build_sample_report()
        assert [s["rank"] for s in report.suggestions] == [1, 2]
        assert report.suggestions[0]["id"] == "dataloader_starvation"
        assert report.suggestions[0]["effort"] == "low"
        assert report.suggestions[1]["id"] == "no_compile"
        assert report.suggestions[1]["expected_gain"] == "Modest speedup"

    def test_empty_findings(self) -> None:
        report = build_sample_report(findings=[])
        assert report.findings == []
        assert report.suggestions == []
        assert report.summary["finding_count"] == 0
        assert report.summary["top_finding"] is None

    def test_run_meta_is_copied(self) -> None:
        meta: Dict[str, Any] = {"device": "cpu"}
        report = build_report(
            run_meta=meta,
            phases=make_phases(),
            gpu=make_gpu(),
            syncs=make_syncs(),
            memory=MemorySummary(),
            findings=[],
        )
        meta["device"] = "mutated"
        assert report.run_meta["device"] == "cpu"


class TestRenderText:
    def test_contains_sections(self) -> None:
        text = render_text(build_sample_report())
        assert "XDL Training Diagnostics Report" in text
        assert "Phase Timing" in text
        assert "GPU / Memory / Syncs" in text
        assert "Findings" in text
        assert "dataloader_starvation" in text
        assert "num_workers=8; pin_memory=True" in text
        assert "No issues detected" not in text

    def test_header_values(self) -> None:
        text = render_text(build_sample_report())
        assert "steps:          50" in text
        assert "cuda:0" in text
        assert "42.5%" in text
        assert "150.0 W" in text

    def test_empty_findings_says_no_issues(self) -> None:
        text = render_text(build_sample_report(findings=[]))
        assert "No issues detected" in text
        assert "Findings" in text

    def test_returns_string_with_trailing_newline(self) -> None:
        text = render_text(build_sample_report())
        assert isinstance(text, str)
        assert text.endswith("\n")


class TestWriteReport:
    def test_writes_both_artifacts(self, tmp_path: Path) -> None:
        report = build_sample_report()
        json_path, txt_path = write_report(str(tmp_path), report)
        assert Path(json_path).name == "diagnostics_report.json"
        assert Path(txt_path).name == "diagnostics_report.txt"
        assert Path(json_path).exists()
        assert Path(txt_path).exists()

    def test_json_round_trip(self, tmp_path: Path) -> None:
        report = build_sample_report()
        json_path, _ = write_report(str(tmp_path), report)
        loaded = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert loaded == report.to_dict()
        assert loaded["schema_version"] == 1
        assert len(loaded["findings"]) == 2
        assert loaded["findings"][0]["id"] == "dataloader_starvation"
        assert loaded["suggestions"][0]["rank"] == 1

    def test_creates_nested_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "reports"
        json_path, txt_path = write_report(str(target), build_sample_report())
        assert Path(json_path).exists()
        assert Path(txt_path).exists()

    def test_txt_matches_render_text(self, tmp_path: Path) -> None:
        report = build_sample_report()
        _, txt_path = write_report(str(tmp_path), report)
        assert Path(txt_path).read_text(encoding="utf-8") == render_text(report)

    def test_utf8_not_escaped(self, tmp_path: Path) -> None:
        report = build_sample_report()
        report.run_meta["note"] = "训练诊断"
        json_path, _ = write_report(str(tmp_path), report)
        raw = Path(json_path).read_text(encoding="utf-8")
        assert "训练诊断" in raw
        assert "\\u" not in raw
