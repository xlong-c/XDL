"""Tests for the rule-based diagnostics analyzer.

All records are synthetic; the tests run on CPU with no GPU and no torch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from xdl.diagnostics.analyzer import (
    AnalysisContext,
    AnalyzerConfig,
    DiagnosticAnalyzer,
)
from xdl.diagnostics.records import (
    GpuSummary,
    MemorySummary,
    PhaseAggregate,
    SyncSummary,
)


def make_phases(
    fracs: Optional[Dict[str, float]] = None,
    steps: int = 100,
) -> PhaseAggregate:
    fracs = dict(fracs or {})
    means = {name: frac for name, frac in fracs.items()}
    means.setdefault("total", 1.0)
    return PhaseAggregate(
        steps=steps,
        means=means,
        p50=dict(means),
        p95=dict(means),
        fracs=fracs,
    )


def make_gpu(
    util_mean: Optional[float] = 50.0,
    available: bool = True,
    mem_used_frac_mean: Optional[float] = 0.8,
) -> GpuSummary:
    return GpuSummary(
        samples=10,
        util_mean=util_mean,
        util_p50=util_mean,
        util_p95=util_mean,
        power_mean=100.0,
        power_max=120.0,
        power_limit=250.0,
        mem_used_frac_mean=mem_used_frac_mean,
        temp_max=70.0,
        available=available,
    )


def make_syncs(
    log_item_per_step: float = 0.0,
    grad_norm_syncs_per_step: float = 0.0,
) -> SyncSummary:
    return SyncSummary(
        steps=100,
        log_item_total=int(log_item_per_step * 100),
        grad_norm_total=int(grad_norm_syncs_per_step * 100),
        empty_cache_events=0,
        log_item_per_step=log_item_per_step,
        grad_norm_syncs_per_step=grad_norm_syncs_per_step,
    )


def make_ctx(
    fracs: Optional[Dict[str, float]] = None,
    gpu: Optional[GpuSummary] = None,
    syncs: Optional[SyncSummary] = None,
    memory: Optional[MemorySummary] = None,
    config: Optional[Dict[str, Any]] = None,
    model: Optional[Dict[str, Any]] = None,
    deep_dive: Optional[Dict[str, Any]] = None,
) -> AnalysisContext:
    return AnalysisContext(
        phases=make_phases(fracs),
        gpu=gpu if gpu is not None else make_gpu(),
        syncs=syncs if syncs is not None else make_syncs(),
        memory=memory if memory is not None else MemorySummary(),
        config=dict(config or {}),
        model=dict(model or {}),
        deep_dive=deep_dive,
    )


def ids(findings) -> list:
    return [f.id for f in findings]


class TestDataloaderStarvation:
    def test_fires_when_data_wait_high_and_util_low(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.3, "train_step": 0.7},
            gpu=make_gpu(util_mean=40.0),
        )
        result = DiagnosticAnalyzer().analyze(ctx)
        assert "dataloader_starvation" in ids(result)

    def test_absent_when_util_high(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.3},
            gpu=make_gpu(util_mean=80.0),
        )
        assert "dataloader_starvation" not in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_below_threshold(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.1},
            gpu=make_gpu(util_mean=20.0),
        )
        assert "dataloader_starvation" not in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_severity_high_above_0_4(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.5},
            gpu=make_gpu(util_mean=10.0),
        )
        finding = next(
            f
            for f in DiagnosticAnalyzer().analyze(ctx)
            if f.id == "dataloader_starvation"
        )
        assert finding.severity == "high"
        assert finding.suggestion is not None
        assert "num_workers" in finding.suggestion.code_hint


class TestNumWorkersZero:
    def test_fires(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.2},
            gpu=make_gpu(util_mean=80.0),
            config={"num_workers": 0},
        )
        assert "num_workers_zero" in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_workers_set(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.2},
            gpu=make_gpu(util_mean=80.0),
            config={"num_workers": 4},
        )
        assert "num_workers_zero" not in ids(DiagnosticAnalyzer().analyze(ctx))


class TestNoAmp:
    def test_fires_for_fp32_conv_model_low_util(self) -> None:
        ctx = make_ctx(
            fracs={"train_step": 0.8},
            gpu=make_gpu(util_mean=50.0),
            config={"precision": "fp32"},
            model={"has_conv_or_linear": True},
        )
        assert "no_amp" in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_amp_enabled(self) -> None:
        ctx = make_ctx(
            fracs={"train_step": 0.8},
            gpu=make_gpu(util_mean=50.0),
            config={"precision": "amp"},
            model={"has_conv_or_linear": True},
        )
        assert "no_amp" not in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_util_high(self) -> None:
        ctx = make_ctx(
            fracs={"train_step": 0.8},
            gpu=make_gpu(util_mean=75.0),
            config={"precision": "fp32"},
            model={"has_conv_or_linear": True},
        )
        assert "no_amp" not in ids(DiagnosticAnalyzer().analyze(ctx))


class TestGradNormSyncStall:
    def test_fires(self) -> None:
        ctx = make_ctx(
            fracs={"nan_monitor": 0.1},
            syncs=make_syncs(grad_norm_syncs_per_step=1.0),
            config={"nan_monitor": True},
        )
        assert "grad_norm_sync_stall" in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_nan_monitor_off(self) -> None:
        ctx = make_ctx(
            fracs={"nan_monitor": 0.1},
            syncs=make_syncs(grad_norm_syncs_per_step=1.0),
            config={"nan_monitor": False},
        )
        assert "grad_norm_sync_stall" not in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_frac_low(self) -> None:
        ctx = make_ctx(
            fracs={"nan_monitor": 0.01},
            syncs=make_syncs(grad_norm_syncs_per_step=1.0),
            config={"nan_monitor": True},
        )
        assert "grad_norm_sync_stall" not in ids(DiagnosticAnalyzer().analyze(ctx))


class TestLogItemSyncStall:
    def test_fires(self) -> None:
        ctx = make_ctx(
            fracs={"callbacks": 0.2},
            syncs=make_syncs(log_item_per_step=5.0),
        )
        assert "log_item_sync_stall" in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_few_items(self) -> None:
        ctx = make_ctx(
            fracs={"callbacks": 0.2},
            syncs=make_syncs(log_item_per_step=1.0),
        )
        assert "log_item_sync_stall" not in ids(DiagnosticAnalyzer().analyze(ctx))


class TestFrozenBackbone:
    def test_fires(self) -> None:
        ctx = make_ctx(
            fracs={"train_step": 0.8},
            model={"frozen_param_frac": 0.6},
        )
        assert "frozen_backbone_recompute" in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_frac_low(self) -> None:
        ctx = make_ctx(
            fracs={"train_step": 0.8},
            model={"frozen_param_frac": 0.1},
        )
        assert "frozen_backbone_recompute" not in ids(DiagnosticAnalyzer().analyze(ctx))


class TestEmptyCacheThrash:
    def test_fires_on_sawtooth(self) -> None:
        ctx = make_ctx(memory=MemorySummary(sawtooth_count=5))
        assert "empty_cache_thrash" in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_fires_on_deep_dive(self) -> None:
        ctx = make_ctx(
            memory=MemorySummary(sawtooth_count=0),
            deep_dive={"empty_cache_events": 2},
        )
        assert "empty_cache_thrash" in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_clean(self) -> None:
        ctx = make_ctx(memory=MemorySummary(sawtooth_count=0))
        assert "empty_cache_thrash" not in ids(DiagnosticAnalyzer().analyze(ctx))


class TestLowVramSmallBatch:
    def test_fires(self) -> None:
        ctx = make_ctx(
            gpu=make_gpu(mem_used_frac_mean=0.2),
            config={"batch_size": 4, "accumulation_steps": 1},
        )
        assert "low_vram_small_batch" in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_accumulating(self) -> None:
        ctx = make_ctx(
            gpu=make_gpu(mem_used_frac_mean=0.2),
            config={"batch_size": 4, "accumulation_steps": 4},
        )
        assert "low_vram_small_batch" not in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_batch_large(self) -> None:
        ctx = make_ctx(
            gpu=make_gpu(mem_used_frac_mean=0.2),
            config={"batch_size": 64},
        )
        assert "low_vram_small_batch" not in ids(DiagnosticAnalyzer().analyze(ctx))


class TestH2dOverhead:
    def test_fires(self) -> None:
        ctx = make_ctx(fracs={"h2d": 0.2})
        result = DiagnosticAnalyzer().analyze(ctx)
        assert "h2d_overhead" in ids(result)
        finding = next(f for f in result if f.id == "h2d_overhead")
        assert finding.suggestion is not None
        assert "pin_memory" in finding.suggestion.code_hint
        assert "non_blocking" in finding.suggestion.code_hint

    def test_absent_below_threshold(self) -> None:
        ctx = make_ctx(fracs={"h2d": 0.05})
        assert "h2d_overhead" not in ids(DiagnosticAnalyzer().analyze(ctx))


class TestNoCompile:
    def test_fires(self) -> None:
        ctx = make_ctx(fracs={"train_step": 0.8}, model={"compiled": False})
        assert "no_compile" in ids(DiagnosticAnalyzer().analyze(ctx))

    def test_absent_when_compiled(self) -> None:
        ctx = make_ctx(fracs={"train_step": 0.8}, model={"compiled": True})
        assert "no_compile" not in ids(DiagnosticAnalyzer().analyze(ctx))


class TestRankingAndScoring:
    def test_sorted_descending_by_score(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.5, "h2d": 0.3, "train_step": 0.2},
            gpu=make_gpu(util_mean=10.0),
            config={"precision": "fp32"},
            model={"has_conv_or_linear": True, "compiled": False},
        )
        result = DiagnosticAnalyzer().analyze(ctx)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_confidence_in_bounds(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.9, "h2d": 0.9, "train_step": 0.9},
            gpu=make_gpu(util_mean=0.0),
            config={"precision": "fp32", "num_workers": 0},
            model={"has_conv_or_linear": True, "compiled": False},
        )
        result = DiagnosticAnalyzer().analyze(ctx)
        assert result
        for finding in result:
            assert 0.0 <= finding.confidence <= 1.0
            assert 0.0 <= finding.attributable_fraction <= 1.0
            assert finding.score >= 0.0

    def test_high_severity_ranks_first(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.6, "h2d": 0.15},
            gpu=make_gpu(util_mean=5.0),
        )
        result = DiagnosticAnalyzer().analyze(ctx)
        assert result[0].id == "dataloader_starvation"

    def test_config_threshold_override(self) -> None:
        strict = DiagnosticAnalyzer(AnalyzerConfig(data_wait_frac_high=0.8))
        ctx = make_ctx(
            fracs={"data_wait": 0.5},
            gpu=make_gpu(util_mean=10.0),
        )
        assert "dataloader_starvation" not in ids(strict.analyze(ctx))


class TestGpuUnavailable:
    def test_no_crash_and_gpu_rules_skipped(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.5, "h2d": 0.3, "train_step": 0.8},
            gpu=GpuSummary(available=False),
            config={"precision": "fp32", "num_workers": 0},
            model={"has_conv_or_linear": True, "compiled": False},
        )
        result = DiagnosticAnalyzer().analyze(ctx)
        result_ids = ids(result)
        assert "dataloader_starvation" not in result_ids
        assert "no_amp" not in result_ids
        assert "num_workers_zero" in result_ids
        assert "h2d_overhead" in result_ids
        assert "no_compile" in result_ids

    def test_none_util_no_crash(self) -> None:
        ctx = make_ctx(
            fracs={"data_wait": 0.5},
            gpu=GpuSummary(available=True, util_mean=None),
        )
        result = DiagnosticAnalyzer().analyze(ctx)
        assert "dataloader_starvation" not in ids(result)

    def test_none_mem_frac_no_crash(self) -> None:
        ctx = make_ctx(
            gpu=make_gpu(mem_used_frac_mean=None),
            config={"batch_size": 2},
        )
        result = DiagnosticAnalyzer().analyze(ctx)
        assert "low_vram_small_batch" not in ids(result)


class TestDefaultConfig:
    def test_default_instance_usable(self) -> None:
        analyzer = DiagnosticAnalyzer()
        assert analyzer.config.data_wait_frac_high == 0.20
        assert analyzer.analyze(make_ctx()) == []
