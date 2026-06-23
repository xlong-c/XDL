"""Single-file XQT detection practice entry.

The default path uses XQT's smoke detection module, synthetic detection data
and the Pythonic ``XQTOptimizationSession`` API. It intentionally does not import
ultralytics and does not embed a full JSON-shaped workflow dict in Python code.

Set XQT_YOLO_PRACTICE_CONFIG to a YAML workflow path when you want to run a
different workflow, for example:

    XQT_YOLO_PRACTICE_CONFIG=xqt/recipes/detection/hf_rtdetr_r18vd_qdq_trt_tensorrt_friendly_eval.yaml \
        python examples/yolo_detection_practice.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xqt import XQTOptimizationSession
from xqt.data import (
    SyntheticDetectionSpec,
    build_synthetic_detection_loader,
)
from xqt.model import build_smoke_detection_module
from xqt.workflows import OptimizedModelResult, optimize_model

import ultralytics
CONFIG_ENV = "XQT_YOLO_PRACTICE_CONFIG"


def _repo_root() -> Path:
    return _REPO_ROOT


def _artifact_path(*parts: str) -> str:
    return str(_repo_root().joinpath("artifacts", "xqt", "detection", *parts))


def _default_detection_loader(
    *,
    sample_limit: int,
    seed: int,
) -> Any:
    return build_synthetic_detection_loader(
        SyntheticDetectionSpec(
            sample_limit=sample_limit,
            batch_size=1,
            image_shape=[3, 64, 64],
            num_classes=3,
            boxes_per_image=2,
            seed=seed,
        )
    )


def _build_default_session() -> XQTOptimizationSession:
    artifact_dir = _artifact_path("yolo_practice_example")
    model = build_smoke_detection_module(
        num_classes=3,
        boxes_per_image=2,
        input_channels=3,
    )
    return XQTOptimizationSession(
        project={
            "name": "yolo_detection_practice_example",
            "artifact_dir": artifact_dir,
        },
        model=model,
        model_config={
            "params": {
                "num_classes": 3,
                "boxes_per_image": 2,
                "input_channels": 3,
            },
            "dtype": "float32",
            "device": "cpu",
        },
        task={
            "type": "detection",
            "class_names": ["class_0", "class_1", "class_2"],
            "detection_postprocess": {
                "format": "yolo_raw",
                "box_format": "xyxy",
                "score_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_detections": 100,
                "score_activation": "sigmoid",
                "has_objectness": False,
                "class_agnostic_nms": False,
            },
            "detection_metric": {
                "iou_thresholds": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "max_detections": 100,
            },
            "params": {"imgsz": 64},
        },
        data_splits={
            "calibration": _default_detection_loader(sample_limit=2, seed=42),
            "validation": _default_detection_loader(sample_limit=4, seed=2),
        },
    )


def _run_default_stages(session: XQTOptimizationSession) -> None:
    session.eval(name="baseline_eval", split="validation", baseline=True)
    session.benchmark(
        name="baseline_latency",
        split="validation",
        warmup=1,
        iterations=2,
    )
    session.export(
        name="export_fp32_onnx",
        split="validation",
        format="onnx",
        output_path=_artifact_path(
            "yolo_practice_example",
            "export_fp32_onnx",
            "smoke_detection_fp32.onnx",
        ),
        opset=18,
        target_params={
            "input_names": ["images"],
            "output_names": ["predictions"],
            "dynamo": False,
            "runtime_diff": False,
        },
    )
    session.runtime_eval(
        name="fp32_onnx_runtime",
        split="validation",
        compare_to="baseline_eval",
        artifact="last_onnx",
        input_names=["images"],
        max_batches=2,
        warmup=0,
        iterations=1,
        accept={
            "metric": "map50_95",
            "max_drop": 0.20,
            "max_mean_abs": 0.05,
            "max_max_abs": 0.5,
        },
    )
    session.quant(
        name="quant_qdq",
        calibration_split="calibration",
        validation_split="validation",
        save_model=False,
        backend="onnxruntime_qdq",
        strategy="static_int8",
        policy={
            "source_name": "smoke_detection_source.onnx",
            "output_path": _artifact_path(
                "yolo_practice_example",
                "quant_qdq",
                "smoke_detection_qdq.onnx",
            ),
            "input_names": ["images"],
            "output_names": ["predictions"],
            "dynamo": False,
            "sample_limit": 2,
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "op_types_to_quantize": ["Conv"],
            "extra_options": {"ActivationSymmetric": False},
        },
    )
    session.runtime_eval(
        name="qdq_onnx_runtime",
        split="validation",
        compare_to="baseline_eval",
        artifact="quant_onnx",
        input_names=["images"],
        max_batches=2,
        warmup=0,
        iterations=1,
        accept={
            "metric": "map50_95",
            "max_drop": 0.20,
            "max_mean_abs": 0.25,
            "max_max_abs": 2.5,
        },
    )
    session.prune(
        name="prune_sparse",
        from_stage="initial",
        split="validation",
        method="global_l1_unstructured",
        target_sparsity=0.2,
    )
    session.eval(
        name="prune_eval",
        split="validation",
        compare_to="baseline_eval",
        baseline=False,
        accept={"metric": "map50_95", "max_drop": 0.20},
    )
    session.benchmark(
        name="prune_latency",
        split="validation",
        compare_to="baseline_latency",
        warmup=1,
        iterations=2,
    )
    session.prune(
        name="structured_prune_guard",
        from_stage="initial",
        split="validation",
        save_model=False,
        method="structured",
        granularity="channel",
        target_sparsity=0.5,
        accept={"min_speedup": 1.01},
    )


def _run_default_session() -> OptimizedModelResult:
    session = _build_default_session()
    _run_default_stages(session)
    return session.result()


def _configured_workflow_path() -> str | None:
    value = os.environ.get(CONFIG_ENV)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return str(path)


def _get_nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _format_float(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return "-"


def _stage_metric_summary(metrics: Mapping[str, Any]) -> str:
    if "eval" in metrics:
        report = metrics["eval"]
        return (
            "mAP50-95="
            f"{_format_float(_get_nested(report, 'metrics', 'map50_95'))}, "
            f"mAP50={_format_float(_get_nested(report, 'metrics', 'map50'))}"
        )
    if "runtime_eval" in metrics:
        report = metrics["runtime_eval"]
        runtime = report.get("runtime", "runtime")
        mean_ms = _get_nested(report, "latency", "mean_ms")
        return (
            f"{runtime}: mAP50-95="
            f"{_format_float(_get_nested(report, 'metrics', 'map50_95'))}, "
            f"latency_ms={_format_float(mean_ms)}"
        )
    if "quant" in metrics:
        report = metrics["quant"]
        return (
            f"backend={report.get('backend', '-')}, "
            f"artifact={report.get('artifact', report.get('output_path', '-'))}"
        )
    if "prune" in metrics:
        report = metrics["prune"]
        state = report.get("execution_state", "applied")
        return (
            f"method={report.get('method', '-')}, "
            f"state={state}, sparsity={_format_float(report.get('sparsity'))}, "
            f"speedup_claimed={report.get('speedup_claimed', '-')}"
        )
    if "benchmark" in metrics:
        report = metrics["benchmark"]
        return f"latency_ms={_format_float(report.get('mean_ms'))}"
    if "export" in metrics:
        artifacts = metrics["export"].get("artifacts", [])
        return f"artifacts={len(artifacts)}"
    return "-"


def _print_result(result: OptimizedModelResult) -> None:
    print(f"project: {result.context.config.project.name}")
    print(f"baseline_stage: {result.baseline_stage or '-'}")
    print(f"best_stage: {result.best_stage or '-'}")
    print("stages:")
    for stage in result.stages:
        status = "accepted" if stage.accepted else "rejected"
        summary = _stage_metric_summary(stage.metrics)
        print(f"  - {stage.name} [{stage.kind}] {status}: {summary}")

    workflow_manifest = result.context.artifacts.get("workflow_manifest")
    if workflow_manifest is not None:
        print(f"workflow_manifest: {workflow_manifest}")
    workflow_result = result.context.artifacts.get("workflow_result")
    if workflow_result is not None:
        print(f"workflow_result: {workflow_result}")


def main() -> int:
    workflow_path = _configured_workflow_path()
    if workflow_path is not None:
        print(f"workflow_config: {workflow_path}")
        result = optimize_model(workflow_path)
    else:
        print("workflow_config: pythonic interactive detection practice")
        result = _run_default_session()
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
