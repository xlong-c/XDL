"""
YOLO detection practice example for XQT.

默认配置:
1. 使用 Ultralytics `yolo11n.pt`
2. 自动下载 `coco8.yaml`
3. 运行 detection scenario matrix,覆盖 baseline / quant / prune / operator / export 组合

简单用法:
- 直接运行完整矩阵:
  `python examples/yolo_detection_practice.py`
- 只运行部分场景:
  `XQT_YOLO_PRACTICE_SCENARIOS=baseline,quant_export python examples/yolo_detection_practice.py`
- 使用自定义 recipe:
  `XQT_YOLO_PRACTICE_CONFIG=/abs/path/to/recipe.yaml python examples/yolo_detection_practice.py`
- 关闭自动 FP16 ONNX 旁路评估:
  `XQT_YOLO_FP16_ONNX=0 python examples/yolo_detection_practice.py`
- 额外导出 Ultralytics 官方参考 ONNX:
  `XQT_YOLO_EXPORT_REFERENCE=1 python examples/yolo_detection_practice.py`
- 调整日志级别:
  `XQT_YOLO_PRACTICE_LOG_LEVEL=DEBUG python examples/yolo_detection_practice.py`
- 调整终端日志级别:
  `XQT_YOLO_PRACTICE_CONSOLE_LOG_LEVEL=INFO python examples/yolo_detection_practice.py`

切换配置:
- 设置 `XQT_YOLO_PRACTICE_CONFIG=/abs/path/to/recipe.yaml`
- 设置 `XQT_YOLO_PRACTICE_SCENARIOS=baseline,quant_only` 只运行部分场景

主要输出:
- 总矩阵: `artifacts/xqt/yolo_detection_practice/scenario_matrix.json`
- 文本摘要: `artifacts/xqt/yolo_detection_practice/scenario_report.txt`
- 运行日志: `artifacts/xqt/yolo_detection_practice/yolo_detection_practice.log`
- 单场景运行时对比: `<artifact_dir>/detection_runtime_scenarios.json`
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # 允许从源码 checkout 直接运行示例,无需先安装 editable package.
    sys.path.insert(0, str(REPO_ROOT))

from xqt.core.artifact import ArtifactRecord, MetricRecord
from xqt.core.config import load_xqt_config
from xqt.eval import evaluate_detection_runtime_model, evaluate_onnx_detection_model
from xqt.export import convert_onnx_to_fp16
from xqt.model import export_ultralytics_reference, resolve_ultralytics_dataset
from xqt.operator_opt import (
    build_operator_optimization_plan,
    materialize_operator_candidate_model,
)
from xqt.pipeline.runner import run_xqt_recipe


CONFIG_ENV = "XQT_YOLO_PRACTICE_CONFIG"
SCENARIOS_ENV = "XQT_YOLO_PRACTICE_SCENARIOS"
EXPORT_REFERENCE_ENV = "XQT_YOLO_EXPORT_REFERENCE"
FP16_ONNX_ENV = "XQT_YOLO_FP16_ONNX"
LOG_LEVEL_ENV = "XQT_YOLO_PRACTICE_LOG_LEVEL"
CONSOLE_LOG_LEVEL_ENV = "XQT_YOLO_PRACTICE_CONSOLE_LOG_LEVEL"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "xqt"
    / "recipes"
    / "yolo_detection_practice.yaml"
)
LOGGER = logging.getLogger("xqt.yolo_detection_practice")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    overrides: dict[str, Any]


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _load_recipe_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, str(DEFAULT_CONFIG_PATH))).expanduser()


def _configure_logging(log_path: Path) -> None:
    raw_level = os.environ.get(LOG_LEVEL_ENV, "INFO").strip().upper()
    level = getattr(logging, raw_level, logging.INFO)
    raw_console_level = os.environ.get(CONSOLE_LOG_LEVEL_ENV, "WARNING").strip().upper()
    console_level = getattr(logging, raw_console_level, logging.WARNING)

    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()

    LOGGER.setLevel(level)
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def _prepare_dataset(config: Any) -> dict[str, Any]:
    dataset_yaml = str(config.task.params.get("dataset_yaml", "coco8.yaml"))
    info = resolve_ultralytics_dataset(dataset_yaml, autodownload=True)
    config.task.class_names = list(info.names)
    return info.to_dict()


def _apply_dataset_info(config: Any, dataset_info: dict[str, Any]) -> None:
    names = dataset_info.get("names")
    if isinstance(names, list):
        config.task.class_names = [str(name) for name in names]


def _validation_loader(context: Any) -> Any:
    loader = context.data.get("validation")
    if loader is None:
        raise ValueError(
            "validation dataloader is required for YOLO detection practice"
        )
    return loader


def _runtime_eval_kwargs(context: Any) -> dict[str, Any]:
    return {
        "postprocess": context.config.task.detection_postprocess,
        "metric_config": context.config.task.detection_metric,
        "device": context.config.model.device,
        "max_batches": context.config.data.validation.sample_limit,
        "atol": context.config.validation.output_diff.atol,
        "rtol": context.config.validation.output_diff.rtol,
        "benchmark_warmup": context.config.benchmark.warmup,
        "benchmark_iterations": context.config.benchmark.iterations,
        "benchmark_sync_cuda": context.config.benchmark.sync_cuda,
    }


def _fp32_onnx_export_path(context: Any) -> str | None:
    export_metrics = context.metrics.get("export")
    if not isinstance(export_metrics, dict):
        return None
    artifacts = export_metrics.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("format") == "onnx" and artifact.get("path") is not None:
            return str(artifact["path"])
    return None


def _qdq_onnx_path(context: Any) -> str | None:
    quant_metrics = context.metrics.get("quant")
    if not isinstance(quant_metrics, dict):
        return None
    path = quant_metrics.get("path")
    if path is not None:
        return str(path)
    artifacts = quant_metrics.get("artifacts")
    if isinstance(artifacts, dict):
        model_artifacts = artifacts.get("model")
        if (
            isinstance(model_artifacts, dict)
            and model_artifacts.get("onnx") is not None
        ):
            return str(model_artifacts["onnx"])
    return None


def _fp16_onnx_artifact(context: Any, fp32_onnx_path: str) -> dict[str, Any] | None:
    if not _env_flag(FP16_ONNX_ENV, default=True):
        return None
    artifact_dir = Path(context.config.project.artifact_dir)
    output_path = artifact_dir / "yolo_fp16.onnx"
    LOGGER.info("Converting FP32 ONNX to FP16: %s -> %s", fp32_onnx_path, output_path)
    try:
        result = convert_onnx_to_fp16(
            fp32_onnx_path,
            output_path,
            keep_io_types=False,
        )
    except Exception as exc:
        LOGGER.warning("FP16 ONNX conversion failed: %s", exc)
        return {
            "error": str(exc),
            "source_path": fp32_onnx_path,
            "path": str(output_path),
        }
    if context.manifest is not None and hasattr(context.manifest, "add_artifact"):
        context.manifest.add_artifact(
            ArtifactRecord(
                path=str(result.path),
                format="onnx",
                runtime="onnxruntime",
                checksum=result.checksum,
                metadata={
                    "precision": "fp16",
                    "kind": "fp16_onnx_baseline",
                    **result.metadata,
                },
            )
        )
    return {
        "path": str(result.path),
        "source_path": fp32_onnx_path,
        "checksum": result.checksum,
        "metadata": dict(result.metadata),
    }


def _scenario_artifact_dir(root_artifact_dir: Path, scenario_name: str) -> Path:
    return root_artifact_dir / "scenarios" / scenario_name


def _scenario_specs(root_artifact_dir: Path) -> list[ScenarioSpec]:
    prune_override = {
        "enabled": True,
        "method": "global_l1_unstructured",
        "target_sparsity": 0.2,
    }
    disabled_export = {"targets": []}
    # 每个场景只声明 recipe overrides,实际执行仍走 XQT 主 pipeline.
    return [
        ScenarioSpec(
            name="baseline",
            description="PyTorch plus FP32/FP16 ONNX Runtime baseline",
            overrides={
                "project": {
                    "name": "yolo_detection_practice_baseline",
                    "artifact_dir": str(
                        _scenario_artifact_dir(root_artifact_dir, "baseline")
                    ),
                },
                "compression": {
                    "axes": [],
                    "quant": {"enabled": False},
                    "prune": {"enabled": False},
                },
                "operator_optimization": {"enabled": False},
                "export": {
                    "targets": [
                        {
                            "format": "onnx",
                            "output_path": str(
                                _scenario_artifact_dir(root_artifact_dir, "baseline")
                                / "yolo11n_fp32.onnx"
                            ),
                            "opset": 18,
                            "params": {
                                "input_names": ["images"],
                                "output_names": ["predictions"],
                                "dynamo": False,
                                "runtime_diff": False,
                            },
                        }
                    ]
                },
            },
        ),
        ScenarioSpec(
            name="quant_only",
            description="ONNX Runtime QDQ INT8 without extra export targets",
            overrides={
                "project": {
                    "name": "yolo_detection_practice_quant_only",
                    "artifact_dir": str(
                        _scenario_artifact_dir(root_artifact_dir, "quant_only")
                    ),
                },
                "compression": {
                    "axes": ["precision"],
                    "quant": {"enabled": True},
                    "prune": {"enabled": False},
                },
                "operator_optimization": {"enabled": False},
                "export": disabled_export,
            },
        ),
        ScenarioSpec(
            name="prune_only",
            description="Global L1 unstructured pruning plus export/parser baseline",
            overrides={
                "project": {
                    "name": "yolo_detection_practice_prune_only",
                    "artifact_dir": str(
                        _scenario_artifact_dir(root_artifact_dir, "prune_only")
                    ),
                },
                "compression": {
                    "axes": ["sparsity"],
                    "quant": {"enabled": False},
                    "prune": prune_override,
                },
                "operator_optimization": {"enabled": False},
                "export": {
                    "targets": [
                        {
                            "format": "onnx",
                            "output_path": str(
                                _scenario_artifact_dir(root_artifact_dir, "prune_only")
                                / "yolo11n_pruned_fp32.onnx"
                            ),
                            "opset": 18,
                            "params": {
                                "input_names": ["images"],
                                "output_names": ["predictions"],
                                "dynamo": False,
                                "runtime_diff": False,
                            },
                        }
                    ]
                },
            },
        ),
        ScenarioSpec(
            name="operator_only",
            description="torch.compile and deployment-target operator validation",
            overrides={
                "project": {
                    "name": "yolo_detection_practice_operator_only",
                    "artifact_dir": str(
                        _scenario_artifact_dir(root_artifact_dir, "operator_only")
                    ),
                },
                "compression": {
                    "axes": [],
                    "quant": {"enabled": False},
                    "prune": {"enabled": False},
                },
                "operator_optimization": {
                    "enabled": True,
                    "targets": [
                        {
                            "name": "model",
                            "backend": "torch_compile",
                            "mode": "reduce-overhead",
                            "fullgraph": False,
                            "min_speedup": 1.01,
                            "validate": {
                                "atol": 5.0e-2,
                                "rtol": 5.0e-2,
                            },
                        },
                        {
                            "name": "backbone",
                            "backend": "deployment_backend",
                            "options": {
                                "runtime": "onnxruntime",
                                "stage": "export_profile",
                                "semantic_target": "backbone",
                            },
                        },
                        {
                            "name": "neck",
                            "backend": "deployment_backend",
                            "options": {
                                "runtime": "onnxruntime",
                                "stage": "export_profile",
                                "semantic_target": "neck",
                            },
                        },
                        {
                            "name": "head",
                            "backend": "deployment_backend",
                            "options": {
                                "runtime": "onnxruntime",
                                "stage": "export_profile",
                                "semantic_target": "head",
                            },
                        },
                        {
                            "name": "postprocess",
                            "backend": "deployment_backend",
                            "options": {
                                "runtime": "external_postprocess",
                                "stage": "decode_nms",
                                "semantic_target": "decode_nms",
                                "fallback_reason": "postprocess optimization is validated through decoded detection diff",
                            },
                        },
                    ],
                },
                "export": disabled_export,
            },
        ),
        ScenarioSpec(
            name="quant_export",
            description="QDQ plus FP32/FP16 ONNX, TensorRT dry-run, and OpenVINO dry-run export",
            overrides={
                "project": {
                    "name": "yolo_detection_practice_quant_export",
                    "artifact_dir": str(
                        _scenario_artifact_dir(root_artifact_dir, "quant_export")
                    ),
                },
                "compression": {
                    "axes": ["precision"],
                    "quant": {"enabled": True},
                    "prune": {"enabled": False},
                },
                "operator_optimization": {"enabled": False},
            },
        ),
        ScenarioSpec(
            name="prune_quant",
            description="Pruning before ONNX Runtime QDQ INT8 with parser dry-runs",
            overrides={
                "project": {
                    "name": "yolo_detection_practice_prune_quant",
                    "artifact_dir": str(
                        _scenario_artifact_dir(root_artifact_dir, "prune_quant")
                    ),
                },
                "compression": {
                    "axes": ["precision", "sparsity"],
                    "quant": {"enabled": True},
                    "prune": prune_override,
                },
                "operator_optimization": {"enabled": False},
            },
        ),
        ScenarioSpec(
            name="full_chain",
            description="Prune + quant + operator/export full chain",
            overrides={
                "project": {
                    "name": "yolo_detection_practice_full_chain",
                    "artifact_dir": str(
                        _scenario_artifact_dir(root_artifact_dir, "full_chain")
                    ),
                },
                "compression": {
                    "axes": ["precision", "sparsity"],
                    "quant": {"enabled": True},
                    "prune": prune_override,
                },
                "operator_optimization": {"enabled": True},
            },
        ),
    ]


def _selected_scenarios(root_artifact_dir: Path) -> list[ScenarioSpec]:
    specs = _scenario_specs(root_artifact_dir)
    raw = os.environ.get(SCENARIOS_ENV)
    if raw is None or not raw.strip():
        return specs
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    selected = [spec for spec in specs if spec.name in requested]
    missing = sorted(set(requested) - {spec.name for spec in selected})
    if missing:
        raise ValueError(f"unknown YOLO practice scenarios: {', '.join(missing)}")
    return selected


def _operator_candidate_runtime(
    context: Any,
    *,
    baseline_model: Any,
) -> dict[str, Any] | None:
    if baseline_model is None:
        return None
    quant_metrics = context.metrics.get("quant")
    if isinstance(quant_metrics, dict):
        backend = str(quant_metrics.get("backend") or "")
        runtime = str(quant_metrics.get("runtime") or "")
        if backend == "onnxruntime_qdq" or runtime == "onnxruntime":
            return None
    plan = build_operator_optimization_plan(context.config.operator_optimization)
    # deployment_backend 目标只用于 manifest/report,这里仅物化可执行的 PyTorch 编译候选.
    executable_targets = [
        target for target in plan.targets if target.backend == "torch_compile"
    ]
    if not executable_targets:
        return None

    candidate_model = baseline_model
    total_compile_time_ms = 0.0
    try:
        for target in executable_targets:
            candidate_model, compile_time_ms = materialize_operator_candidate_model(
                candidate_model,
                target,
            )
            if compile_time_ms is not None:
                total_compile_time_ms += float(compile_time_ms)
    except Exception as exc:
        return {
            "runtime": "pytorch_compiled_candidate",
            "error": str(exc),
            "targets": [target.to_dict() for target in executable_targets],
        }

    runtime_report = evaluate_detection_runtime_model(
        candidate_model,
        _validation_loader(context),
        reference_model=baseline_model,
        runtime="pytorch_compiled_candidate",
        **_runtime_eval_kwargs(context),
    ).to_dict()
    runtime_report["compile_time_ms"] = total_compile_time_ms
    runtime_report["targets"] = [target.to_dict() for target in executable_targets]
    return runtime_report


def _metric_drop_summary(scenarios: dict[str, Any]) -> dict[str, Any]:
    baseline_metrics = scenarios.get("baseline_pytorch", {}).get("metrics")
    if not isinstance(baseline_metrics, dict):
        return {}
    summary: dict[str, Any] = {}
    for runtime_name, runtime_report in scenarios.items():
        metrics = (
            runtime_report.get("metrics") if isinstance(runtime_report, dict) else None
        )
        if not isinstance(metrics, dict):
            continue
        runtime_summary: dict[str, Any] = {}
        for metric_name, baseline_value in baseline_metrics.items():
            candidate_value = metrics.get(metric_name)
            if not isinstance(baseline_value, (int, float)) or not isinstance(
                candidate_value,
                (int, float),
            ):
                continue
            runtime_summary[f"{metric_name}_drop"] = float(baseline_value) - float(
                candidate_value
            )
        if runtime_summary:
            summary[runtime_name] = runtime_summary
    return summary


def _latency_speedup_summary(scenarios: dict[str, Any]) -> dict[str, Any]:
    baseline_latency = scenarios.get("baseline_pytorch", {}).get("latency")
    if not isinstance(baseline_latency, dict):
        return {}
    baseline_mean = baseline_latency.get("mean_ms")
    if not isinstance(baseline_mean, (int, float)) or float(baseline_mean) <= 0.0:
        return {}
    summary: dict[str, Any] = {}
    for runtime_name, runtime_report in scenarios.items():
        latency = (
            runtime_report.get("latency") if isinstance(runtime_report, dict) else None
        )
        if not isinstance(latency, dict):
            continue
        mean_ms = latency.get("mean_ms")
        if not isinstance(mean_ms, (int, float)) or float(mean_ms) <= 0.0:
            continue
        summary[runtime_name] = {
            "mean_ms": float(mean_ms),
            "speedup_vs_baseline_pytorch": float(baseline_mean) / float(mean_ms),
        }
    return summary


def _record_runtime_diff_metrics(
    context: Any,
    *,
    scenario_name: str,
    runtime_name: str,
    runtime_report: dict[str, Any],
) -> None:
    if context.manifest is None or not hasattr(context.manifest, "add_metric"):
        return
    for diff_name in ("raw_output_diff", "decoded_diff"):
        diff_payload = runtime_report.get(diff_name)
        if not isinstance(diff_payload, dict):
            continue
        for metric_name in (
            "max_abs",
            "mean_abs",
            "box_mae",
            "score_mae",
            "label_match_rate",
        ):
            value = diff_payload.get(metric_name)
            if isinstance(value, (int, float, bool)):
                context.manifest.add_metric(
                    MetricRecord(
                        name=f"runtime.{runtime_name}.{diff_name}.{metric_name}",
                        value=value,
                        metadata={"scenario": scenario_name},
                    )
                )


def _collect_runtime_scenarios(
    context: Any,
    *,
    scenario_name: str,
) -> dict[str, Any]:
    # recipe pass 先产出模型和导出物,这里再对可执行 runtime 做统一评估.
    LOGGER.info("Collecting runtime evaluations for scenario %s", scenario_name)
    validation_loader = _validation_loader(context)
    baseline_model = (
        context.reference_model
        if context.reference_model is not None
        else context.model
    )
    current_model = context.model
    scenarios: dict[str, Any] = {}
    if baseline_model is not None:
        LOGGER.info("Evaluating %s baseline_pytorch runtime", scenario_name)
        scenarios["baseline_pytorch"] = evaluate_detection_runtime_model(
            baseline_model,
            validation_loader,
            reference_model=baseline_model,
            runtime="pytorch",
            **_runtime_eval_kwargs(context),
        ).to_dict()
    if current_model is not None:
        LOGGER.info("Evaluating %s current_pytorch runtime", scenario_name)
        scenarios["current_pytorch"] = evaluate_detection_runtime_model(
            current_model,
            validation_loader,
            reference_model=baseline_model,
            runtime="pytorch",
            **_runtime_eval_kwargs(context),
        ).to_dict()
    if context.config.operator_optimization.enabled:
        operator_candidate = _operator_candidate_runtime(
            context,
            baseline_model=baseline_model,
        )
        if operator_candidate is not None:
            scenarios["operator_candidate_pytorch"] = operator_candidate

    fp32_onnx_path = _fp32_onnx_export_path(context)
    if fp32_onnx_path is not None:
        LOGGER.info(
            "Evaluating %s fp32_onnx runtime: %s", scenario_name, fp32_onnx_path
        )
        scenarios["fp32_onnx"] = evaluate_onnx_detection_model(
            fp32_onnx_path,
            validation_loader,
            input_names=["images"],
            reference_model=current_model,
            **_runtime_eval_kwargs(context),
        ).to_dict()
        fp16_onnx = _fp16_onnx_artifact(context, fp32_onnx_path)
        if fp16_onnx is not None:
            fp16_onnx_path = fp16_onnx.get("path")
            if "error" in fp16_onnx:
                scenarios["fp16_onnx"] = {
                    "runtime": "onnxruntime",
                    "error": fp16_onnx["error"],
                    "metadata": fp16_onnx,
                }
            elif fp16_onnx_path is not None:
                LOGGER.info(
                    "Evaluating %s fp16_onnx runtime: %s",
                    scenario_name,
                    fp16_onnx_path,
                )
                scenarios["fp16_onnx"] = evaluate_onnx_detection_model(
                    str(fp16_onnx_path),
                    validation_loader,
                    input_names=["images"],
                    reference_model=current_model,
                    **_runtime_eval_kwargs(context),
                ).to_dict()
    quant_onnx_path = _qdq_onnx_path(context)
    if quant_onnx_path is not None:
        LOGGER.info(
            "Evaluating %s quant_onnx_qdq runtime: %s", scenario_name, quant_onnx_path
        )
        scenarios["quant_onnx_qdq"] = evaluate_onnx_detection_model(
            quant_onnx_path,
            validation_loader,
            input_names=["images"],
            reference_model=current_model,
            **_runtime_eval_kwargs(context),
        ).to_dict()

    scenario_path = (
        Path(context.config.project.artifact_dir) / "detection_runtime_scenarios.json"
    )
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(
        json.dumps(scenarios, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s runtime scenario JSON: %s", scenario_name, scenario_path)

    if context.manifest is not None and hasattr(context.manifest, "add_artifact"):
        context.manifest.add_artifact(
            ArtifactRecord.from_file(
                scenario_path,
                format="json",
                runtime="pytorch",
                metadata={
                    "kind": "detection_runtime_scenarios",
                    "scenario": scenario_name,
                },
            )
        )
    if context.manifest is not None and hasattr(context.manifest, "add_metric"):
        for runtime_name, runtime_report in scenarios.items():
            metrics = runtime_report.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                context.manifest.add_metric(
                    MetricRecord(
                        name=f"runtime.{runtime_name}.{metric_name}",
                        value=value,
                        metadata={"scenario": scenario_name},
                    )
                )
            _record_runtime_diff_metrics(
                context,
                scenario_name=scenario_name,
                runtime_name=runtime_name,
                runtime_report=runtime_report,
            )
        metric_drops = _metric_drop_summary(scenarios)
        latency_speedups = _latency_speedup_summary(scenarios)
        for runtime_name, drops in metric_drops.items():
            for metric_name, value in drops.items():
                context.manifest.add_metric(
                    MetricRecord(
                        name=f"runtime.{runtime_name}.{metric_name}",
                        value=value,
                        metadata={"scenario": scenario_name},
                    )
                )
        for runtime_name, speedup in latency_speedups.items():
            context.manifest.add_metric(
                MetricRecord(
                    name=f"runtime.{runtime_name}.speedup_vs_baseline_pytorch",
                    value=speedup["speedup_vs_baseline_pytorch"],
                    metadata={
                        "scenario": scenario_name,
                        "mean_ms": speedup["mean_ms"],
                    },
                )
            )
    if context.manifest is not None and hasattr(context.manifest, "write_json"):
        manifest_path = Path(context.config.project.artifact_dir) / "manifest.json"
        context.manifest.write_json(manifest_path)
        context.artifacts["manifest"] = manifest_path

    return {
        "scenarios": scenarios,
        "path": str(scenario_path),
        "metric_drops": _metric_drop_summary(scenarios),
        "latency_speedups": _latency_speedup_summary(scenarios),
    }


def _scenario_summary(
    context: Any,
    *,
    scenario_name: str,
    scenario_description: str,
) -> dict[str, Any]:
    runtime_scenarios = _collect_runtime_scenarios(
        context,
        scenario_name=scenario_name,
    )
    manifest_path = Path(context.config.project.artifact_dir) / "manifest.json"
    return {
        "name": scenario_name,
        "description": scenario_description,
        "project": context.config.project.name,
        "artifact_dir": context.config.project.artifact_dir,
        "compression_axes": list(context.config.compression.axes),
        "passes": context.manifest.passes if context.manifest is not None else [],
        "baseline": context.metrics.get("baseline"),
        "prune": context.metrics.get("prune"),
        "quant": context.metrics.get("quant"),
        "export": context.metrics.get("export"),
        "benchmark": context.metrics.get("benchmark"),
        "operator_optimization": context.metrics.get("operator_optimization"),
        "runtime": runtime_scenarios,
        "metrics": {
            "drops": runtime_scenarios["metric_drops"],
            "latency_speedups": runtime_scenarios["latency_speedups"],
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "runtime_scenarios": runtime_scenarios["path"],
        },
        "runtime_scenarios": runtime_scenarios["scenarios"],
        "runtime_scenarios_path": runtime_scenarios["path"],
        "manifest_path": str(manifest_path),
    }


def _run_scenario(
    config_path: Path,
    dataset_info: dict[str, Any],
    scenario: ScenarioSpec,
) -> dict[str, Any]:
    LOGGER.info("Starting scenario %s: %s", scenario.name, scenario.description)
    config = load_xqt_config(config_path, overrides=scenario.overrides)
    _apply_dataset_info(config, dataset_info)
    context = run_xqt_recipe(config)
    summary = _scenario_summary(
        context,
        scenario_name=scenario.name,
        scenario_description=scenario.description,
    )
    LOGGER.info(
        "Finished scenario %s; artifacts: %s", scenario.name, summary["artifact_dir"]
    )
    return summary


def _maybe_export_ultralytics_reference(
    config: Any,
    artifact_dir: Path,
) -> dict[str, Any] | None:
    if not _env_flag(EXPORT_REFERENCE_ENV, default=False):
        return None
    dataset_yaml = str(config.task.params.get("dataset_yaml", "coco8.yaml"))
    imgsz = int(config.task.params.get("imgsz", 640))
    calibration_fraction = float(config.task.params.get("calibration_fraction", 0.5))
    LOGGER.info("Exporting Ultralytics reference ONNX into %s", artifact_dir)
    export_path = export_ultralytics_reference(
        str(config.model.params.get("weights", "yolo11n.pt")),
        format="onnx",
        imgsz=imgsz,
        data=dataset_yaml,
        fraction=calibration_fraction,
        int8=True,
        dynamic=False,
        nms=False,
        project=str(artifact_dir),
        name="ultralytics_reference",
    )
    return {
        "path": str(export_path),
        "dataset_yaml": dataset_yaml,
        "imgsz": imgsz,
        "calibration_fraction": calibration_fraction,
    }


def _matrix_summary(scenarios: dict[str, Any]) -> dict[str, Any]:
    # 顶层 summary 只做索引,详细指标保留在各场景 manifest 和 runtime sidecar 中.
    summary: dict[str, Any] = {
        "scenario_count": len(scenarios),
        "runtime_coverage": {},
        "quantized_op_types": [],
        "calibration": {},
        "export_formats": {},
        "operator_fallbacks": {},
        "prune": {},
    }
    quantized_op_types: set[str] = set()
    for scenario_name, scenario in scenarios.items():
        runtime_scenarios = scenario.get("runtime_scenarios", {})
        if isinstance(runtime_scenarios, dict):
            summary["runtime_coverage"][scenario_name] = sorted(
                runtime_scenarios.keys()
            )
        quant = scenario.get("quant")
        if isinstance(quant, dict):
            metadata = quant.get("metadata")
            if isinstance(metadata, dict):
                for op_type in metadata.get("quantized_op_types", []) or []:
                    quantized_op_types.add(str(op_type))
                calibration_summary = metadata.get("calibration_summary")
                if isinstance(calibration_summary, dict):
                    summary["calibration"][scenario_name] = calibration_summary
        export = scenario.get("export")
        if isinstance(export, dict):
            artifacts = export.get("artifacts")
            if isinstance(artifacts, list):
                summary["export_formats"][scenario_name] = [
                    {
                        "format": item.get("format"),
                        "dry_run": item.get("dry_run"),
                        "precision": item.get("precision"),
                        "profiles": item.get("profiles"),
                        "path": item.get("path"),
                    }
                    for item in artifacts
                    if isinstance(item, dict)
                ]
        operator = scenario.get("operator_optimization")
        if isinstance(operator, dict):
            targets = operator.get("targets")
            if isinstance(targets, list):
                summary["operator_fallbacks"][scenario_name] = [
                    {
                        "target": item.get("target_name"),
                        "backend": item.get("backend"),
                        "applied": item.get("applied"),
                        "skip_reason": item.get("skip_reason"),
                        "metadata": item.get("metadata"),
                    }
                    for item in targets
                    if isinstance(item, dict)
                ]
        prune = scenario.get("prune")
        if isinstance(prune, dict):
            summary["prune"][scenario_name] = {
                "method": prune.get("method"),
                "sparsity": prune.get("sparsity"),
                "parameter_sparsity": prune.get("parameter_sparsity"),
                "export_status": prune.get("export_status"),
                "benchmark_status": prune.get("benchmark_status"),
            }
    summary["quantized_op_types"] = sorted(quantized_op_types)
    return summary


def _text_cell(value: Any, *, max_length: int = 180) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = f"{value:.6g}"
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > max_length:
        return f"{text[: max_length - 3]}..."
    return text


def _table_lines(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["(none)"]
    rendered_rows = [[_text_cell(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered_rows))
        for index, header in enumerate(headers)
    ]
    lines = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rendered_rows:
        lines.append(
            " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return lines


def _runtime_metric_rows(scenarios: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for scenario_name, scenario in scenarios.items():
        if not isinstance(scenario, dict):
            continue
        runtime_scenarios = scenario.get("runtime_scenarios")
        if not isinstance(runtime_scenarios, dict):
            continue
        metric_summary = scenario.get("metrics")
        drops = (
            metric_summary.get("drops", {}) if isinstance(metric_summary, dict) else {}
        )
        speedups = (
            metric_summary.get("latency_speedups", {})
            if isinstance(metric_summary, dict)
            else {}
        )
        for runtime_name, runtime_report in runtime_scenarios.items():
            if not isinstance(runtime_report, dict):
                continue
            latency = runtime_report.get("latency")
            runtime_speedup = (
                speedups.get(runtime_name) if isinstance(speedups, dict) else None
            )
            rows.append(
                [
                    scenario_name,
                    runtime_name,
                    runtime_report.get("metrics"),
                    latency.get("mean_ms") if isinstance(latency, dict) else None,
                    (
                        runtime_speedup.get("speedup_vs_baseline_pytorch")
                        if isinstance(runtime_speedup, dict)
                        else None
                    ),
                    drops.get(runtime_name) if isinstance(drops, dict) else None,
                    runtime_report.get("error") or "ok",
                ]
            )
    return rows


def _scenario_index_rows(scenarios: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for scenario_name, scenario in scenarios.items():
        if not isinstance(scenario, dict):
            continue
        runtime_scenarios = scenario.get("runtime_scenarios")
        rows.append(
            [
                scenario_name,
                scenario.get("compression_axes"),
                sorted(runtime_scenarios.keys())
                if isinstance(runtime_scenarios, dict)
                else [],
                scenario.get("artifact_dir"),
            ]
        )
    return rows


def _quant_rows(scenarios: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for scenario_name, scenario in scenarios.items():
        quant = scenario.get("quant") if isinstance(scenario, dict) else None
        if not isinstance(quant, dict):
            continue
        metadata = quant.get("metadata")
        rows.append(
            [
                scenario_name,
                quant.get("backend") or quant.get("runtime"),
                quant.get("path"),
                metadata.get("quantized_op_types")
                if isinstance(metadata, dict)
                else None,
                (
                    metadata.get("calibration_summary")
                    if isinstance(metadata, dict)
                    else None
                ),
            ]
        )
    return rows


def _export_rows(scenarios: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for scenario_name, scenario in scenarios.items():
        export = scenario.get("export") if isinstance(scenario, dict) else None
        if not isinstance(export, dict):
            continue
        artifacts = export.get("artifacts")
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            rows.append(
                [
                    scenario_name,
                    artifact.get("format"),
                    artifact.get("precision"),
                    artifact.get("dry_run"),
                    artifact.get("path"),
                ]
            )
    return rows


def _operator_rows(scenarios: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for scenario_name, scenario in scenarios.items():
        operator = (
            scenario.get("operator_optimization")
            if isinstance(scenario, dict)
            else None
        )
        if not isinstance(operator, dict):
            continue
        targets = operator.get("targets")
        if not isinstance(targets, list):
            continue
        for target in targets:
            if not isinstance(target, dict):
                continue
            rows.append(
                [
                    scenario_name,
                    target.get("target_name"),
                    target.get("backend"),
                    target.get("applied"),
                    target.get("skip_reason"),
                ]
            )
    return rows


def _prune_rows(scenarios: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for scenario_name, scenario in scenarios.items():
        prune = scenario.get("prune") if isinstance(scenario, dict) else None
        if not isinstance(prune, dict):
            continue
        rows.append(
            [
                scenario_name,
                prune.get("method"),
                prune.get("sparsity"),
                prune.get("parameter_sparsity"),
                prune.get("export_status"),
                prune.get("benchmark_status"),
            ]
        )
    return rows


def _build_text_report(
    output: dict[str, Any],
    *,
    report_path: Path,
    log_path: Path,
) -> str:
    dataset = output.get("dataset")
    dataset_yaml = dataset.get("yaml_path") if isinstance(dataset, dict) else None
    dataset_root = dataset.get("root") if isinstance(dataset, dict) else None
    scenarios = output.get("scenarios")
    scenario_map = scenarios if isinstance(scenarios, dict) else {}

    lines = [
        "YOLO Detection Practice Report",
        "=" * 30,
        "",
        f"Project: {_text_cell(output.get('project'))}",
        f"Config: {_text_cell(output.get('config_path'))}",
        f"Artifact dir: {_text_cell(output.get('artifact_dir'))}",
        f"Dataset: {_text_cell(dataset_yaml)}",
        f"Dataset root: {_text_cell(dataset_root)}",
        f"Scenario count: {_text_cell(len(scenario_map))}",
        f"Matrix JSON: {_text_cell(output.get('matrix_path'))}",
        f"Report TXT: {_text_cell(report_path)}",
        f"Log file: {_text_cell(log_path)}",
        "",
        "Scenario Index",
        "-" * 14,
        *_table_lines(
            ["scenario", "axes", "runtimes", "artifact_dir"],
            _scenario_index_rows(scenario_map),
        ),
        "",
        "Runtime Metrics",
        "-" * 15,
        *_table_lines(
            ["scenario", "runtime", "metrics", "mean_ms", "speedup", "drops", "status"],
            _runtime_metric_rows(scenario_map),
        ),
        "",
        "Quantization",
        "-" * 12,
        *_table_lines(
            ["scenario", "backend", "path", "quantized_ops", "calibration"],
            _quant_rows(scenario_map),
        ),
        "",
        "Exports",
        "-" * 7,
        *_table_lines(
            ["scenario", "format", "precision", "dry_run", "path"],
            _export_rows(scenario_map),
        ),
        "",
        "Operator Optimization",
        "-" * 21,
        *_table_lines(
            ["scenario", "target", "backend", "applied", "skip_reason"],
            _operator_rows(scenario_map),
        ),
        "",
        "Pruning",
        "-" * 7,
        *_table_lines(
            [
                "scenario",
                "method",
                "sparsity",
                "parameter_sparsity",
                "export_status",
                "benchmark_status",
            ],
            _prune_rows(scenario_map),
        ),
        "",
        "Full structured data is in scenario_matrix.json.",
        "",
    ]
    return "\n".join(lines)


def _write_text_report(
    output: dict[str, Any],
    *,
    report_path: Path,
    log_path: Path,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _build_text_report(output, report_path=report_path, log_path=log_path),
        encoding="utf-8",
    )
    return report_path


def _console_summary(output: dict[str, Any]) -> dict[str, Any]:
    scenario_order = output.get("scenario_order")
    return {
        "project": output.get("project"),
        "scenario_count": len(scenario_order)
        if isinstance(scenario_order, list)
        else 0,
        "scenario_order": scenario_order,
        "outputs": output.get("outputs"),
    }


def main() -> int:
    config_path = _load_recipe_path()
    base_config = load_xqt_config(config_path)
    root_artifact_dir = Path(base_config.project.artifact_dir)
    root_artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = root_artifact_dir / "yolo_detection_practice.log"
    _configure_logging(log_path)
    LOGGER.info("Loaded YOLO detection practice config: %s", config_path)
    dataset_info = _prepare_dataset(base_config)
    LOGGER.info("Resolved dataset: %s", dataset_info.get("yaml_path") or dataset_info)

    scenarios: dict[str, Any] = {}
    selected_scenarios = _selected_scenarios(root_artifact_dir)
    LOGGER.info(
        "Selected scenarios: %s",
        ", ".join(scenario.name for scenario in selected_scenarios),
    )
    for scenario in selected_scenarios:
        scenarios[scenario.name] = _run_scenario(
            config_path,
            dataset_info,
            scenario,
        )

    matrix_path = root_artifact_dir / "scenario_matrix.json"
    report_path = root_artifact_dir / "scenario_report.txt"
    output: dict[str, Any] = {
        "project": base_config.project.name,
        "config_path": str(config_path),
        "artifact_dir": str(root_artifact_dir),
        "dataset": dataset_info,
        "scenario_order": list(scenarios.keys()),
        "scenarios": scenarios,
        "summary": _matrix_summary(scenarios),
        "matrix_path": str(matrix_path),
        "outputs": {
            "matrix_json": str(matrix_path),
            "text_report": str(report_path),
            "log": str(log_path),
        },
    }
    reference_export = _maybe_export_ultralytics_reference(
        base_config,
        root_artifact_dir,
    )
    if reference_export is not None:
        output["ultralytics_reference_export"] = reference_export

    matrix_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_text_report(output, report_path=report_path, log_path=log_path)
    LOGGER.info("Wrote scenario matrix JSON: %s", matrix_path)
    LOGGER.info("Wrote text report: %s", report_path)
    print(
        json.dumps(
            _console_summary(output), indent=2, ensure_ascii=False, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
