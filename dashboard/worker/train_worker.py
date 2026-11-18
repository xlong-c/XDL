"""Launch one XDL training run from a dashboard-generated manifest.

manifest 有两种形态:

- 模板 run: 带 ``template_path``, 由 ``build_template_config()`` 应用 allowlist
  覆盖后走 ``setup_from_yaml()``.
- workflow run: 带 ``kind == "workflow"``, 先校验快照哈希, 再调用 workflow compiler
  编译, 最后按执行 profile 走 schema v1 或 dag-runtime.

两种形态共用同一套 ``Trainer`` 生命周期, 控制文件和事件通道.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from xdl.callbacks import DashboardCallback
from xdl.config import TrainSetup, setup_from_yaml
from xdl.trainer import Trainer, TrainerControl
from xdl.workflow import WorkflowError

from .config_builder import build_template_config
from .control_file import ControlFileWatcher
from .event_sink import EventSink
from .workflow_compiler import CompileOptions, compile_workflow
from .workflow_model import (
    COMPILER_VERSION,
    WORKFLOW_MANIFEST_KIND,
    load_workflow_file,
)

logger = logging.getLogger("xdl.dashboard.worker")
MANIFEST_ENV = "XDL_DASHBOARD_MANIFEST"


def _load_manifest(path: str) -> Dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("worker manifest must be a JSON object")
    return value


def is_workflow_manifest(manifest: Mapping[str, Any]) -> bool:
    """判断 manifest 是否描述一个 workflow run."""
    return manifest.get("kind") == WORKFLOW_MANIFEST_KIND


def run_worker(manifest: Dict[str, Any], event_sink: EventSink) -> int:
    """Build XDL setup and execute the configured training lifecycle."""
    config_path = Path(str(manifest["config_path"])).expanduser().resolve()
    control_path = Path(str(manifest["control_path"])).expanduser().resolve()
    control = TrainerControl()
    if int(os.environ.get("RANK", "0")) == 0:
        control.set_observer(
            lambda state, sequence: event_sink.emit(
                "run_state",
                {"state": state, "control_sequence": sequence},
                True,
            )
        )
    watcher = ControlFileWatcher(control_path, control)
    watcher.start()

    try:
        event_sink.emit("run_state", {"state": "validating"}, reliable=True)
        setup = build_setup(manifest, config_path, event_sink)
        setup.callbacks.append(
            DashboardCallback(
                event_sink.emit,
                every_n_steps=int(manifest.get("metrics_every_n_steps", 1)),
                interval_s=float(manifest.get("metrics_interval_s", 2.0)),
            )
        )
        trainer = Trainer.from_setup(setup)
        model = setup.create_model()
        resume_checkpoint = manifest.get("resume_checkpoint") or None
        trainer.fit(
            model,
            setup.train_loader,
            setup.val_loader,
            control=control,
            resume_from_checkpoint=(
                str(resume_checkpoint) if resume_checkpoint is not None else None
            ),
        )
        checkpoint_path = None
        for callback in trainer.callbacks:
            candidate = getattr(callback, "last_model_path", None)
            if candidate:
                checkpoint_path = str(candidate)
                break
        event_sink.emit(
            "worker_exit",
            {
                "exit_code": 0,
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "control_state": control.state,
                "stop_mode": control.stop_mode,
                "checkpoint_requested": control.save_checkpoint_requested,
                "checkpoint_path": checkpoint_path,
            },
            reliable=True,
        )
        return 0
    except BaseException as exc:
        event_sink.emit(
            "worker_error",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            reliable=True,
        )
        raise
    finally:
        watcher.stop()


def build_setup(
    manifest: Mapping[str, Any], config_path: Path, event_sink: EventSink
) -> TrainSetup:
    """按 manifest 形态构建 ``TrainSetup``."""
    if is_workflow_manifest(manifest):
        return _build_workflow_setup(manifest, config_path, event_sink)
    template_path = manifest.get("template_path")
    if template_path:
        raw_overrides = manifest.get("overrides", {})
        raw_allowed_paths = manifest.get("allowed_paths", [])
        raw_system_overrides = manifest.get("system_overrides", {})
        if (
            not isinstance(raw_overrides, dict)
            or not isinstance(raw_allowed_paths, list)
            or not isinstance(raw_system_overrides, dict)
        ):
            raise ValueError("template manifest config fields must be containers")
        resolved_config = build_template_config(
            template_path,
            raw_overrides,
            config_path,
            allowed_paths=[str(path) for path in raw_allowed_paths],
            system_overrides=raw_system_overrides,
            use_accelerate=bool(manifest.get("use_accelerate", False)),
        )
        return setup_from_yaml(resolved_config)
    return setup_from_yaml(config_path.resolve(strict=True))


def _build_workflow_setup(
    manifest: Mapping[str, Any], config_path: Path, event_sink: EventSink
) -> TrainSetup:
    loaded = load_workflow_file(
        str(manifest["workflow_path"]),
        expected_hash=_optional_text(manifest.get("workflow_snapshot_hash")),
    )
    options = CompileOptions(
        data_root=_optional_text(manifest.get("data_root")),
        project_root=_optional_text(manifest.get("project_root")),
        output_path=str(config_path),
        system_overrides=_system_overrides(manifest.get("system_overrides")),
        check_registry=True,
    )
    result = compile_workflow(loaded.payload, options)
    if not result.ok:
        _emit_compile_failure(event_sink, result)
        raise WorkflowError(_summarize_issues(result.issues))
    event_sink.emit(
        "workflow_compiled",
        {
            "compiler_version": result.compiler_version,
            "execution_profile": result.execution_profile,
            "config_path": result.config_path,
            "config_hash": result.config_hash,
            "snapshot_hash": loaded.digest,
        },
        reliable=True,
    )
    if result.config_path:
        return setup_from_yaml(result.config_path)
    from xdl.workflow.executor import WorkflowExecutor

    executor = WorkflowExecutor()
    return executor.build_setup(
        loaded.snapshot,
        system_overrides=_system_overrides(manifest.get("system_overrides")),
    )


def _emit_compile_failure(event_sink: EventSink, result: Any) -> None:
    issues = [issue.as_dict() for issue in result.issues]
    message = _summarize_issues(result.issues)
    event_sink.emit(
        "run_error",
        {
            "stage": "compile",
            "compiler_version": COMPILER_VERSION,
            "execution_profile": result.execution_profile,
            "message": message,
            "issues": issues,
        },
        reliable=True,
    )


def _summarize_issues(issues: Any) -> str:
    parts = [f"{issue.code}: {issue.message}" for issue in issues]
    return "; ".join(parts) or "workflow compilation failed"


def _system_overrides(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def main() -> int:
    """Run the worker using only the manifest path from the environment."""
    manifest_path = os.environ.get(MANIFEST_ENV)
    if not manifest_path:
        logger.error("%s must name the worker manifest", MANIFEST_ENV)
        return 2

    event_stream = sys.stdout
    sink = EventSink()
    writer = threading.Thread(
        target=sink.run,
        kwargs={"stream": event_stream},
        name="xdl-dashboard-event-writer",
        daemon=True,
    )
    writer.start()

    exit_code = 0
    try:
        manifest = _load_manifest(manifest_path)
        # Trainer and user components may print freely; stdout remains a clean
        # JSONL event protocol for the Go parent process.
        with contextlib.redirect_stdout(sys.stderr):
            exit_code = run_worker(manifest, sink)
    except BaseException as exc:
        sink.emit(
            "worker_error",
            {"error_type": type(exc).__name__, "message": str(exc)},
            reliable=True,
        )
        traceback.print_exc(file=sys.stderr)
        exit_code = 1
    finally:
        sink.close()
        writer.join(timeout=5.0)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
