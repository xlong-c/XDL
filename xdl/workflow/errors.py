"""Workflow IR 的错误码, 校验问题与异常.

错误码是前端, Go dashboard 和 Python compiler 共享的稳定契约. 新增错误码必须
同时更新 ``docs/md/architecture/dashboard-workflow-ir.md`` 的错误码表.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

# 结构层错误码: Go dashboard 与 Python IR 都会产生.
INVALID_SNAPSHOT = "invalid_snapshot"
UNSUPPORTED_VERSION = "unsupported_version"
DUPLICATE_ID = "duplicate_id"
UNKNOWN_NODE = "unknown_node"
UNKNOWN_PORT = "unknown_port"
MISSING_PORT = "missing_port"
DUPLICATE_EDGE = "duplicate_edge"
PORT_CONFLICT = "port_conflict"
DANGLING_OUTPUT = "dangling_output"
SELF_LOOP = "self_loop"
CYCLE_DETECTED = "cycle_detected"
LIMIT_EXCEEDED = "limit_exceeded"
INVALID_PARAM = "invalid_param"

# 语义层错误码: 只有 Python compiler 产生.
UNKNOWN_NODE_KIND = "unknown_node_kind"
UNKNOWN_TARGET = "unknown_target"
UNSUPPORTED_PROFILE = "unsupported_profile"
TYPE_MISMATCH = "type_mismatch"
UNREACHABLE_NODE = "unreachable_node"
MISSING_REQUIRED_NODE = "missing_required_node"
DUPLICATE_ROLE = "duplicate_role"
PATH_NOT_ALLOWED = "path_not_allowed"
UNSUPPORTED_GRAPH = "unsupported_graph"

ERROR_CODES = frozenset(
    {
        INVALID_SNAPSHOT,
        UNSUPPORTED_VERSION,
        DUPLICATE_ID,
        UNKNOWN_NODE,
        UNKNOWN_PORT,
        MISSING_PORT,
        DUPLICATE_EDGE,
        PORT_CONFLICT,
        DANGLING_OUTPUT,
        SELF_LOOP,
        CYCLE_DETECTED,
        LIMIT_EXCEEDED,
        INVALID_PARAM,
        UNKNOWN_NODE_KIND,
        UNKNOWN_TARGET,
        UNSUPPORTED_PROFILE,
        TYPE_MISMATCH,
        UNREACHABLE_NODE,
        MISSING_REQUIRED_NODE,
        DUPLICATE_ROLE,
        PATH_NOT_ALLOWED,
        UNSUPPORTED_GRAPH,
    }
)

LEVEL_ERROR = "error"
LEVEL_WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """一条可定位的校验问题.

    ``path`` 使用 JSON Pointer 风格的斜杠路径 (例如 ``edges/2/to/port``),
    ``node_id`` 与 ``edge_id`` 供前端在画布上直接定位元素.
    """

    code: str
    message: str
    level: str = LEVEL_ERROR
    path: str = ""
    node_id: str = ""
    edge_id: str = ""

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown workflow issue code: {self.code}")
        if self.level not in (LEVEL_ERROR, LEVEL_WARNING):
            raise ValueError(f"unknown workflow issue level: {self.level}")

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "level": self.level,
            "message": self.message,
        }
        if self.path:
            payload["path"] = self.path
        if self.node_id:
            payload["node_id"] = self.node_id
        if self.edge_id:
            payload["edge_id"] = self.edge_id
        return payload


def error_issue(
    code: str,
    message: str,
    *,
    path: str = "",
    node_id: str = "",
    edge_id: str = "",
) -> Issue:
    """构造 error 级问题."""
    return Issue(
        code=code,
        message=message,
        level=LEVEL_ERROR,
        path=path,
        node_id=node_id,
        edge_id=edge_id,
    )


def warning_issue(
    code: str,
    message: str,
    *,
    path: str = "",
    node_id: str = "",
    edge_id: str = "",
) -> Issue:
    """构造 warning 级问题."""
    return Issue(
        code=code,
        message=message,
        level=LEVEL_WARNING,
        path=path,
        node_id=node_id,
        edge_id=edge_id,
    )


def has_errors(issues: Iterable[Issue]) -> bool:
    """判断问题列表中是否存在阻断错误."""
    return any(issue.level == LEVEL_ERROR for issue in issues)


def issues_to_dicts(issues: Sequence[Issue]) -> List[Dict[str, Any]]:
    """把问题列表转换为可直接 JSON 序列化的字典列表."""
    return [issue.as_dict() for issue in issues]


class WorkflowError(Exception):
    """workflow 相关异常基类."""


class WorkflowValidationError(WorkflowError):
    """工作流快照或节点图未通过校验."""

    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues: List[Issue] = list(issues)
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
        super().__init__(detail or "workflow validation failed")

    def as_dicts(self) -> List[Dict[str, Any]]:
        """返回问题字典列表, 便于 CLI 与测试直接消费."""
        return issues_to_dicts(self.issues)


class WorkflowExecutionError(WorkflowError):
    """dag-runtime 执行阶段失败."""


__all__ = [
    "ERROR_CODES",
    "LEVEL_ERROR",
    "LEVEL_WARNING",
    "Issue",
    "WorkflowError",
    "WorkflowExecutionError",
    "WorkflowValidationError",
    "error_issue",
    "has_errors",
    "issues_to_dicts",
    "warning_issue",
]
