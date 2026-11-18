"""XDL workflow: 有类型 DAG 的 IR, 节点目录与 dag-runtime 执行器.

本子包只描述"受节点目录约束的有类型 DAG", 不执行浏览器传来的任意 Python 代码.
标准训练图仍然走 ``xdl.config.setup_from_yaml()``; 只有超出固定训练结构的图才会
进入 :mod:`xdl.workflow.executor`.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .catalog import (
    NodeCatalog,
    NodeKindSpec,
    ParamSpec,
    PortSpec,
    TargetSpec,
    WorkflowLimits,
    load_catalog,
)
from .errors import (
    ERROR_CODES,
    Issue,
    WorkflowError,
    WorkflowExecutionError,
    WorkflowValidationError,
    has_errors,
    issues_to_dicts,
)
from .ir import (
    DEFAULT_PROFILE,
    EXECUTION_PROFILES,
    SNAPSHOT_VERSION,
    PortRef,
    WorkflowEdge,
    WorkflowNode,
    WorkflowSnapshot,
    hash_snapshot_bytes,
    load_snapshot_file,
    parse_snapshot,
    validate_structure,
)

if TYPE_CHECKING:
    from .executor import WorkflowExecutor, WorkflowRuntimeResult

_LAZY_EXPORTS = {"WorkflowExecutor", "WorkflowRuntimeResult"}


def __getattr__(name: str) -> Any:
    """按需导入 dag-runtime 执行器, 避免 IR 使用方被迫加载 torch."""
    if name in _LAZY_EXPORTS:
        from . import executor

        return getattr(executor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_PROFILE",
    "ERROR_CODES",
    "EXECUTION_PROFILES",
    "SNAPSHOT_VERSION",
    "Issue",
    "NodeCatalog",
    "NodeKindSpec",
    "ParamSpec",
    "PortRef",
    "PortSpec",
    "TargetSpec",
    "WorkflowEdge",
    "WorkflowError",
    "WorkflowExecutionError",
    "WorkflowExecutor",
    "WorkflowLimits",
    "WorkflowNode",
    "WorkflowRuntimeResult",
    "WorkflowSnapshot",
    "WorkflowValidationError",
    "has_errors",
    "hash_snapshot_bytes",
    "issues_to_dicts",
    "load_catalog",
    "load_snapshot_file",
    "parse_snapshot",
    "validate_structure",
]
