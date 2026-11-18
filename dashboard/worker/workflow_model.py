"""Worker 侧的工作流快照装载与编译结果模型.

这一层只处理 dashboard worker 关心的 I/O 与结果封装: 读快照文件, 校验哈希,
把编译结果表达成可 JSON 序列化的结构. 图语义与结构校验复用 ``xdl.workflow``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from xdl.workflow import (
    Issue,
    WorkflowError,
    has_errors,
    issues_to_dicts,
    parse_snapshot,
    validate_structure,
)
from xdl.workflow.ir import WorkflowSnapshot

COMPILER_VERSION = "workflow-compiler/1"
WORKFLOW_MANIFEST_KIND = "workflow"
SNAPSHOT_FILENAME = "workflow.json"
GENERATED_CONFIG_FILENAME = "config.yaml"


class SnapshotIntegrityError(WorkflowError):
    """快照文件字节与 manifest 记录的哈希不一致."""


@dataclass(frozen=True)
class LoadedSnapshot:
    """从磁盘读到的快照与其原始字节哈希."""

    path: Path
    raw: bytes
    digest: str
    payload: Mapping[str, Any]
    snapshot: WorkflowSnapshot

    @property
    def execution_profile(self) -> str:
        """返回快照声明的执行 profile."""
        return self.snapshot.execution_profile


def load_workflow_file(
    path: str | Path, *, expected_hash: Optional[str] = None
) -> LoadedSnapshot:
    """读取快照文件, 校验结构与可选哈希.

    ``expected_hash`` 来自 run manifest, 是 Go 写入 canonical JSON 时记录的
    SHA-256. 字节不一致说明快照被改动或写坏, 必须在训练启动前失败.
    """
    resolved = Path(path).expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash and expected_hash != digest:
        raise SnapshotIntegrityError(
            f"workflow snapshot hash mismatch: manifest={expected_hash} "
            f"file={digest} path={resolved}"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError(
            f"workflow snapshot is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SnapshotIntegrityError("workflow snapshot must be a JSON object")
    issues = validate_structure(payload)
    if has_errors(issues):
        raise SnapshotIntegrityError(
            "workflow snapshot failed structural validation: "
            + "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        )
    return LoadedSnapshot(
        path=resolved,
        raw=raw,
        digest=digest,
        payload=payload,
        snapshot=parse_snapshot(payload),
    )


@dataclass(frozen=True)
class CompileResult:
    """编译结果, 可直接 JSON 序列化给 Go 或 worker."""

    ok: bool
    action: str
    execution_profile: str
    issues: Tuple[Issue, ...] = ()
    warnings: Tuple[Issue, ...] = ()
    config: Optional[Mapping[str, Any]] = None
    config_path: str = ""
    config_hash: str = ""
    snapshot_hash: str = ""
    compiler_version: str = COMPILER_VERSION
    runtime_manifest: Mapping[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> Tuple[Issue, ...]:
        """返回 error 级问题."""
        return tuple(issue for issue in self.issues if issue.level == "error")

    def as_dict(self) -> Dict[str, Any]:
        """转换为 CLI 响应字典."""
        payload: Dict[str, Any] = {
            "ok": self.ok,
            "action": self.action,
            "compiler_version": self.compiler_version,
            "execution_profile": self.execution_profile,
            "issues": issues_to_dicts(self.issues),
            "warnings": issues_to_dicts(self.warnings),
        }
        if self.snapshot_hash:
            payload["snapshot_hash"] = self.snapshot_hash
        if self.config is not None:
            payload["config"] = dict(self.config)
        if self.config_path:
            payload["config_path"] = self.config_path
        if self.config_hash:
            payload["config_hash"] = self.config_hash
        if self.runtime_manifest:
            payload["runtime_manifest"] = dict(self.runtime_manifest)
        return payload


def payload_for_exception(exc: BaseException, *, action: str = "") -> Dict[str, Any]:
    """把任意异常映射成协议响应, 保证 stdout 始终是结构化 JSON."""
    from xdl.workflow import WorkflowValidationError

    if isinstance(exc, WorkflowValidationError):
        return {
            "ok": False,
            "action": action,
            "compiler_version": COMPILER_VERSION,
            "issues": exc.as_dicts(),
            "warnings": [],
        }
    return {
        "ok": False,
        "action": action,
        "compiler_version": COMPILER_VERSION,
        "issues": [
            {
                "code": "invalid_snapshot",
                "level": "error",
                "message": f"{type(exc).__name__}: {exc}",
            }
        ],
        "warnings": [],
    }


def config_bytes_hash(payload: bytes) -> str:
    """返回配置文件字节的 SHA-256."""
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "COMPILER_VERSION",
    "GENERATED_CONFIG_FILENAME",
    "SNAPSHOT_FILENAME",
    "WORKFLOW_MANIFEST_KIND",
    "CompileResult",
    "LoadedSnapshot",
    "SnapshotIntegrityError",
    "config_bytes_hash",
    "load_workflow_file",
    "payload_for_exception",
]
