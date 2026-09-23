"""Rule-based diagnostics analyzer.

The analyzer maps measured signals (phase shares, GPU utilization, sync
counters, memory sawtooth) onto ranked root-cause findings. It is pure logic:
no torch import, no device access, no I/O. That keeps it unit-testable on CPU
with synthetic records.

Scoring model::

    score = severity_weight * confidence * attributable_fraction

where ``severity_weight`` is high=3, medium=2, low=1 and ``attributable_fraction``
is the wall-time share of the phase the rule explains, clamped to [0, 1].
Findings are returned sorted by descending score, then by ascending effort
(low first), then by id.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .records import (
    Finding,
    GpuSummary,
    MemorySummary,
    PhaseAggregate,
    Suggestion,
    SyncSummary,
)

_SEVERITY_WEIGHT: Dict[str, int] = {"high": 3, "medium": 2, "low": 1}
_EFFORT_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2}
_FP32_PRECISIONS = (None, "32", "fp32", "float32")


@dataclass
class AnalyzerConfig:
    """Thresholds controlling when each rule fires."""

    data_wait_frac_high: float = 0.20
    gpu_util_low: float = 60.0
    gpu_util_very_low: float = 30.0
    callbacks_frac_high: float = 0.05
    nan_monitor_frac_high: float = 0.05
    h2d_frac_high: float = 0.10
    log_item_per_step_high: float = 3.0
    mem_used_frac_low: float = 0.5
    sawtooth_count_high: int = 3
    frozen_param_frac_high: float = 0.30
    small_batch_threshold: int = 8
    train_step_frac_high: float = 0.5


@dataclass
class AnalysisContext:
    """All measured inputs a single analysis pass operates on."""

    phases: PhaseAggregate
    gpu: GpuSummary
    syncs: SyncSummary
    memory: MemorySummary
    config: Dict[str, Any]
    model: Dict[str, Any]
    deep_dive: Optional[Dict[str, Any]] = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ratio_confidence(value: float, threshold: float, base: float = 0.5) -> float:
    """Confidence rises from ``base`` at the threshold toward 1.0 at 2x it."""
    if threshold <= 0.0:
        return 1.0
    ratio = value / threshold
    if ratio <= 1.0:
        return _clamp(base)
    span = (ratio - 1.0) / 1.0
    return _clamp(base + (1.0 - base) * span)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _finalize(finding: Finding) -> Finding:
    """Fill score from severity/confidence/attributable_fraction."""
    weight = _SEVERITY_WEIGHT.get(finding.severity, 1)
    attr = _clamp(finding.attributable_fraction)
    confidence = _clamp(finding.confidence)
    finding.attributable_fraction = attr
    finding.confidence = confidence
    finding.score = weight * confidence * attr
    return finding


class DiagnosticAnalyzer:
    """Applies all rules to an :class:`AnalysisContext` and ranks findings."""

    def __init__(self, config: AnalyzerConfig = AnalyzerConfig()) -> None:
        self.config = config

    def analyze(self, ctx: AnalysisContext) -> List[Finding]:
        cfg = self.config
        findings: List[Finding] = []
        for rule in (
            self._rule_dataloader_starvation,
            self._rule_num_workers_zero,
            self._rule_no_amp,
            self._rule_grad_norm_sync_stall,
            self._rule_log_item_sync_stall,
            self._rule_frozen_backbone_recompute,
            self._rule_empty_cache_thrash,
            self._rule_low_vram_small_batch,
            self._rule_h2d_overhead,
            self._rule_no_compile,
        ):
            finding = rule(ctx, cfg)
            if finding is not None:
                findings.append(_finalize(finding))
        findings.sort(
            key=lambda f: (
                -f.score,
                _EFFORT_RANK.get(f.suggestion.effort if f.suggestion else "medium", 1),
                f.id,
            )
        )
        return findings

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _rule_dataloader_starvation(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        frac = ctx.phases.frac("data_wait")
        util = ctx.gpu.util_mean
        if frac < cfg.data_wait_frac_high or util is None or util >= cfg.gpu_util_low:
            return None

        severity = "high" if frac >= 0.4 else "medium"
        confidence = _mean(
            [
                _ratio_confidence(frac, cfg.data_wait_frac_high, base=0.6),
                _ratio_confidence(cfg.gpu_util_low - util, cfg.gpu_util_low * 0.5, 0.5),
            ]
        )
        workers = min(8, os.cpu_count() or 1)
        return Finding(
            id="dataloader_starvation",
            severity=severity,
            confidence=confidence,
            title="DataLoader starvation: GPU is waiting on input",
            signals={
                "data_wait_frac": round(frac, 4),
                "gpu_util_mean": round(float(util), 2),
            },
            evidence=[
                f"data_wait consumes {frac:.1%} of step wall time "
                f"(threshold {cfg.data_wait_frac_high:.0%})",
                f"GPU utilization mean is {util:.1f}% (below {cfg.gpu_util_low:.0f}%)",
            ],
            suggestion=Suggestion(
                title="Feed the GPU faster",
                detail=(
                    "Increase DataLoader parallelism and overlap host-to-device "
                    "copies: more workers, pinned memory, persistent workers, "
                    "deeper prefetch, and non_blocking transfers."
                ),
                code_hint=(
                    "DataLoader(dataset, batch_size=B, num_workers=min(8, "
                    "os.cpu_count()), pin_memory=True, persistent_workers=True, "
                    "prefetch_factor=4); use tensor.to(device, non_blocking=True) "
                    "in _transfer_to_device; consider a larger batch."
                ),
                expected_gain="Recover most of the data_wait wall time",
                effort="low",
            ),
            attributable_fraction=frac,
        )

    def _rule_num_workers_zero(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        num_workers = ctx.config.get("num_workers")
        frac = ctx.phases.frac("data_wait")
        if num_workers != 0 or frac <= 0.10:
            return None

        severity = "high" if frac >= 0.4 else "medium"
        confidence = _ratio_confidence(frac, 0.10, base=0.6)
        return Finding(
            id="num_workers_zero",
            severity=severity,
            confidence=confidence,
            title="DataLoader num_workers=0 serializes input loading",
            signals={"num_workers": 0.0, "data_wait_frac": round(frac, 4)},
            evidence=[
                "config num_workers == 0, so loading runs in the main process",
                f"data_wait is {frac:.1%} of step wall time (above 10%)",
            ],
            suggestion=Suggestion(
                title="Enable worker processes",
                detail=(
                    "Set num_workers >= 4, pin_memory=True and "
                    "persistent_workers=True so preprocessing overlaps the "
                    "train step instead of blocking it."
                ),
                code_hint=(
                    "DataLoader(..., num_workers=min(8, os.cpu_count()), "
                    "pin_memory=True, persistent_workers=True, prefetch_factor=4)"
                ),
                expected_gain="Overlap input pipeline with GPU compute",
                effort="low",
            ),
            attributable_fraction=frac,
        )

    def _rule_no_amp(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        precision = ctx.config.get("precision")
        util = ctx.gpu.util_mean
        if precision not in _FP32_PRECISIONS:
            return None
        if not ctx.gpu.available or util is None or util >= 70.0:
            return None
        if not ctx.model.get("has_conv_or_linear"):
            return None

        margin = _clamp((70.0 - util) / 70.0)
        confidence = _clamp(0.5 + 0.4 * margin)
        return Finding(
            id="no_amp",
            severity="medium",
            confidence=confidence,
            title="Mixed precision is disabled on a conv/linear model",
            signals={
                "precision_is_fp32": 1.0,
                "gpu_util_mean": round(float(util), 2),
            },
            evidence=[
                f"precision is {precision!r} (fp32 path)",
                f"GPU utilization mean is {util:.1f}% (< 70%)",
                "model contains conv/linear layers that benefit from AMP",
            ],
            suggestion=Suggestion(
                title="Enable automatic mixed precision",
                detail=(
                    "Wrap the forward/backward in torch.autocast and use a "
                    "GradScaler. fp16/bf16 roughly halves activation memory and "
                    "speeds up matmul/conv on tensor cores."
                ),
                code_hint=(
                    "with torch.autocast(device_type='cuda', dtype=torch.float16): "
                    "loss = model(batch); scaler.scale(loss).backward()"
                ),
                expected_gain="Higher throughput and lower activation memory",
                effort="medium",
            ),
            attributable_fraction=_clamp(1.0 - ctx.phases.frac("data_wait")),
        )

    def _rule_grad_norm_sync_stall(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        if not ctx.config.get("nan_monitor"):
            return None
        if ctx.syncs.grad_norm_syncs_per_step <= 0:
            return None
        frac = ctx.phases.frac("nan_monitor")
        if frac <= cfg.nan_monitor_frac_high:
            return None

        confidence = _ratio_confidence(frac, cfg.nan_monitor_frac_high, base=0.6)
        return Finding(
            id="grad_norm_sync_stall",
            severity="medium",
            confidence=confidence,
            title="NaN-monitor grad-norm sync stalls every step",
            signals={
                "nan_monitor": True,
                "grad_norm_syncs_per_step": ctx.syncs.grad_norm_syncs_per_step,
                "nan_monitor_frac": round(frac, 4),
            },
            evidence=[
                f"nan_monitor is enabled and performs "
                f"{ctx.syncs.grad_norm_syncs_per_step:.2f} D2H grad-norm "
                "syncs per step",
                f"nan_monitor phase is {frac:.1%} of step wall time "
                f"(threshold {cfg.nan_monitor_frac_high:.0%})",
            ],
            suggestion=Suggestion(
                title="Make NaN checks asynchronous or less frequent",
                detail=(
                    "The per-step grad-norm .item() forces a device sync. Check "
                    "every N steps, use a non_blocking copy, or gate the check "
                    "behind a running counter instead of every optimizer step."
                ),
                code_hint=(
                    "if step % 50 == 0: "
                    "gn = torch.nn.utils.clip_grad_norm_(params, max_norm)"
                ),
                expected_gain="Remove one full device sync per step",
                effort="low",
            ),
            attributable_fraction=frac,
        )

    def _rule_log_item_sync_stall(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        per_step = ctx.syncs.log_item_per_step
        if per_step < cfg.log_item_per_step_high:
            return None
        frac = ctx.phases.frac("callbacks")
        if frac <= cfg.callbacks_frac_high:
            return None

        confidence = _mean(
            [
                _ratio_confidence(per_step, cfg.log_item_per_step_high, base=0.55),
                _ratio_confidence(frac, cfg.callbacks_frac_high, base=0.55),
            ]
        )
        return Finding(
            id="log_item_sync_stall",
            severity="medium",
            confidence=confidence,
            title="Per-step .item() logging forces device syncs",
            signals={
                "log_item_per_step": per_step,
                "callbacks_frac": round(frac, 4),
            },
            evidence=[
                f"{per_step:.2f} log tensor .item() calls per step "
                f"(threshold {cfg.log_item_per_step_high:.1f})",
                f"callbacks phase is {frac:.1%} of step wall time "
                f"(threshold {cfg.callbacks_frac_high:.0%})",
            ],
            suggestion=Suggestion(
                title="Batch or detach logging metrics",
                detail=(
                    "Each .item() on a CUDA tensor synchronizes the device. "
                    "Accumulate scalars on device and only materialize them "
                    "every N steps, or log detached tensors to a writer that "
                    "handles sync off the critical path."
                ),
                code_hint=(
                    "self.log('loss', loss.detach(), prefix='train')  # log "
                    "tensor; call .item() only every N steps"
                ),
                expected_gain="Cut per-step sync count and callback overhead",
                effort="medium",
            ),
            attributable_fraction=frac,
        )

    def _rule_frozen_backbone_recompute(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        frozen = ctx.model.get("frozen_param_frac")
        if frozen is None or frozen <= cfg.frozen_param_frac_high:
            return None
        frac = ctx.phases.frac("train_step")
        if frac <= cfg.train_step_frac_high:
            return None

        confidence = _mean(
            [
                _ratio_confidence(frozen, cfg.frozen_param_frac_high, base=0.55),
                _ratio_confidence(frac, cfg.train_step_frac_high, base=0.55),
            ]
        )
        return Finding(
            id="frozen_backbone_recompute",
            severity="medium",
            confidence=confidence,
            title="Frozen backbone still participates in the training graph",
            signals={
                "frozen_param_frac": round(float(frozen), 4),
                "train_step_frac": round(frac, 4),
            },
            evidence=[
                f"{frozen:.0%} of parameters are frozen "
                f"(threshold {cfg.frozen_param_frac_high:.0%})",
                f"train_step is {frac:.1%} of step wall time "
                f"(threshold {cfg.train_step_frac_high:.0%})",
            ],
            suggestion=Suggestion(
                title="Run the frozen backbone under no_grad",
                detail=(
                    "If the frozen portion is only needed for features, compute "
                    "it under torch.no_grad (or cache features) so autograd does "
                    "not build and discard a graph for those parameters."
                ),
                code_hint=(
                    "with torch.no_grad(): features = backbone(x); "
                    "features = features.detach()"
                ),
                expected_gain="Reduce activation memory and backward time",
                effort="medium",
            ),
            attributable_fraction=frac,
        )

    def _rule_empty_cache_thrash(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        sawtooth = ctx.memory.sawtooth_count
        deep_events = 0
        if ctx.deep_dive:
            deep_events = int(ctx.deep_dive.get("empty_cache_events", 0) or 0)
        if sawtooth < cfg.sawtooth_count_high and deep_events <= 0:
            return None

        if sawtooth >= cfg.sawtooth_count_high:
            confidence = _ratio_confidence(
                float(sawtooth), float(cfg.sawtooth_count_high), base=0.6
            )
        else:
            confidence = _clamp(0.6 + 0.1 * min(deep_events, 4))
        return Finding(
            id="empty_cache_thrash",
            severity="medium",
            confidence=confidence,
            title="Repeated empty_cache calls defeat the caching allocator",
            signals={
                "sawtooth_count": float(sawtooth),
                "empty_cache_events": float(deep_events),
                "reserved_peak": float(ctx.memory.reserved_peak),
            },
            evidence=[
                f"reserved-memory sawtooth count is {sawtooth} "
                f"(threshold {cfg.sawtooth_count_high})",
                f"deep dive observed {deep_events} empty_cache events",
            ],
            suggestion=Suggestion(
                title="Stop calling torch.cuda.empty_cache() in the loop",
                detail=(
                    "Periodic empty_cache frees cached blocks and forces the "
                    "allocator to re-request them, adding cudaMalloc stalls and "
                    "fragmenting memory. Remove it from the hot path; call it "
                    "only at run boundaries if at all."
                ),
                code_hint="remove torch.cuda.empty_cache() from training_step",
                expected_gain="Avoid allocator churn and cudaMalloc stalls",
                effort="low",
            ),
            attributable_fraction=1.0,
        )

    def _rule_low_vram_small_batch(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        mem_frac = ctx.gpu.mem_used_frac_mean
        if mem_frac is None or mem_frac >= cfg.mem_used_frac_low:
            return None
        batch_size = ctx.config.get("batch_size")
        if batch_size is None or batch_size >= cfg.small_batch_threshold:
            return None
        if ctx.config.get("accumulation_steps", 1) > 1:
            return None

        confidence = _mean(
            [
                _ratio_confidence(
                    cfg.mem_used_frac_low - mem_frac, cfg.mem_used_frac_low, 0.55
                ),
                _ratio_confidence(
                    float(cfg.small_batch_threshold - batch_size),
                    float(cfg.small_batch_threshold),
                    0.55,
                ),
            ]
        )
        return Finding(
            id="low_vram_small_batch",
            severity="medium",
            confidence=confidence,
            title="Batch is small while VRAM sits mostly idle",
            signals={
                "mem_used_frac_mean": round(float(mem_frac), 4),
                "batch_size": float(batch_size),
                "accumulation_steps": float(ctx.config.get("accumulation_steps", 1)),
            },
            evidence=[
                f"GPU memory used is {mem_frac:.1%} "
                f"(threshold {cfg.mem_used_frac_low:.0%})",
                f"batch_size is {batch_size} (threshold {cfg.small_batch_threshold})",
                "no gradient accumulation is configured",
            ],
            suggestion=Suggestion(
                title="Raise the batch size to fill available VRAM",
                detail=(
                    "Underutilized memory means the GPU is starved per launch. "
                    "Increase batch_size until memory use is ~80%, or add "
                    "gradient accumulation if the effective batch must stay."
                ),
                code_hint="batch_size = 32  # tune upward to ~80% VRAM",
                expected_gain="Better GPU occupancy per kernel launch",
                effort="medium",
            ),
            attributable_fraction=1.0,
        )

    def _rule_h2d_overhead(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        frac = ctx.phases.frac("h2d")
        if frac <= cfg.h2d_frac_high:
            return None

        confidence = _ratio_confidence(frac, cfg.h2d_frac_high, base=0.6)
        return Finding(
            id="h2d_overhead",
            severity="medium",
            confidence=confidence,
            title="Host-to-device copies are not overlapped",
            signals={"h2d_frac": round(frac, 4)},
            evidence=[
                f"h2d copies are {frac:.1%} of step wall time "
                f"(threshold {cfg.h2d_frac_high:.0%})",
                "pageable memory forces synchronous copies",
            ],
            suggestion=Suggestion(
                title="Pin memory and copy asynchronously (coupled fix)",
                detail=(
                    "pin_memory and non_blocking must be applied together: "
                    "pinning enables a DMA engine, and non_blocking lets the "
                    "copy overlap compute. Applying only one gives no benefit."
                ),
                code_hint=(
                    "DataLoader(..., pin_memory=True); "
                    "batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}"
                ),
                expected_gain="Hide H2D transfer behind the train step",
                effort="low",
            ),
            attributable_fraction=frac,
        )

    def _rule_no_compile(
        self, ctx: AnalysisContext, cfg: AnalyzerConfig
    ) -> Optional[Finding]:
        if ctx.model.get("compiled"):
            return None
        frac = ctx.phases.frac("train_step")
        if frac <= cfg.train_step_frac_high:
            return None

        confidence = _ratio_confidence(frac, cfg.train_step_frac_high, base=0.45)
        return Finding(
            id="no_compile",
            severity="low",
            confidence=confidence,
            title="Model is not compiled and the train step dominates",
            signals={"compiled": 0.0, "train_step_frac": round(frac, 4)},
            evidence=[
                "model.compiled is falsy",
                f"train_step is {frac:.1%} of step wall time "
                f"(threshold {cfg.train_step_frac_high:.0%})",
            ],
            suggestion=Suggestion(
                title="Try torch.compile on the train step",
                detail=(
                    "torch.compile can fuse pointwise ops and reduce kernel "
                    "launch overhead. Validate with warmup iterations before "
                    "adopting, and keep an eager fallback."
                ),
                code_hint="model = torch.compile(model)",
                expected_gain="Modest train-step speedup via kernel fusion",
                effort="medium",
            ),
            attributable_fraction=frac,
        )
