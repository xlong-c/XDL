"""节点目录访问与 target 解析.

把 UI 展示用的 label 和真正可构建的组件 target 分开: 目录里的 ``target`` 是唯一
可执行标识, 这里负责把 ``registry:``, ``<module>:`` 和 ``builtin:`` 三种前缀解析到
实际对象, 并给出可定位的校验问题.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional

from xdl.workflow import NodeCatalog, load_catalog
from xdl.workflow.catalog import NodeKindSpec, TargetSpec
from xdl.workflow.errors import Issue, error_issue
from xdl.workflow.errors import UNKNOWN_TARGET

BUILTIN_PREFIX = "builtin:"
REGISTRY_PREFIX = "registry:"

_REGISTRY_BY_KIND = {
    "model": "MODEL_REGISTRY",
    "dataset": "DATASET_REGISTRY",
    "loss": "LOSS_REGISTRY",
    "optimizer": "OPTIMIZER_REGISTRY",
    "scheduler": "SCHEDULER_REGISTRY",
    "metric": "METRIC_REGISTRY",
    "transform": "TRANSFORM_REGISTRY",
    "collate": "COLLATE_REGISTRY",
}


class TargetResolutionError(Exception):
    """target 无法解析到实际组件."""


def load_workflow_catalog(path: Optional[str | Path] = None) -> NodeCatalog:
    """加载节点目录, 默认使用 ``xdl.workflow`` 包内 JSON."""
    return load_catalog(path)


def is_builtin_target(target: str) -> bool:
    """判断 target 是否为 dashboard / runtime 内建节点."""
    return target.startswith(BUILTIN_PREFIX)


def builtin_name(target: str) -> str:
    """取出 ``builtin:`` 前缀后的名称."""
    return target[len(BUILTIN_PREFIX) :] if is_builtin_target(target) else ""


def resolve_target_component(kind: str, target: str) -> Any:
    """把 target 解析为可调用对象.

    ``builtin:`` 由 compiler / executor 处理, 这里返回 ``None``.
    """
    if is_builtin_target(target):
        return None
    if ":" not in target:
        raise TargetResolutionError(f"target {target!r} 必须使用 'source:name' 格式")
    source, name = target.rsplit(":", 1)
    if not source or not name:
        raise TargetResolutionError(f"target {target!r} 必须使用 'source:name' 格式")
    if source == "registry":
        registry = _registry_for_kind(kind)
        try:
            return registry.get(name)
        except Exception as exc:  # noqa: BLE001 - registry 抛出多种错误类型
            raise TargetResolutionError(
                f"{kind} registry 中没有 {name!r}: {exc}"
            ) from exc
    try:
        module = importlib.import_module(source)
    except ImportError as exc:
        raise TargetResolutionError(f"无法导入 target 模块 {source!r}: {exc}") from exc
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise TargetResolutionError(f"模块 {source!r} 没有属性 {name!r}") from exc


def check_target_available(
    node_id: str, kind: str, target: str, *, path: str = ""
) -> Optional[Issue]:
    """校验 target 能在当前环境解析; 成功返回 ``None``."""
    if is_builtin_target(target):
        return None
    try:
        resolve_target_component(kind, target)
    except TargetResolutionError as exc:
        return error_issue(
            UNKNOWN_TARGET,
            f"节点 {node_id} 的 target {target!r} 无法解析: {exc}",
            path=path or f"nodes/{node_id}/target",
            node_id=node_id,
        )
    return None


def resolve_data_path(value: str, data_root: Optional[str | Path]) -> str:
    """把相对数据路径解析到 configured data root 下.

    目录层只保证相对路径与无 ``..``; 这里再做一次解析后的越界检查, 返回绝对路径.
    """
    candidate = Path(value)
    if candidate.is_absolute():
        raise TargetResolutionError(f"数据路径必须是相对路径: {value!r}")
    if data_root is None:
        return str(candidate)
    root = Path(data_root).expanduser().resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TargetResolutionError(f"数据路径逃出 data root: {value!r}") from exc
    return str(resolved)


def _registry_for_kind(kind: str) -> Any:
    from xdl.utils import registry as registry_module

    attribute = _REGISTRY_BY_KIND.get(kind)
    if attribute is None:
        raise TargetResolutionError(f"节点类型 {kind!r} 不支持 registry target")
    return getattr(registry_module, attribute)


def describe_target(kind_spec: NodeKindSpec, target_spec: TargetSpec) -> Dict[str, Any]:
    """返回给前端使用的 target 描述 (label 与可执行 target 分离)."""
    return {
        "kind": kind_spec.kind,
        "target": target_spec.target,
        "label": target_spec.label,
        "summary": target_spec.summary,
        "params": [
            {
                "name": param.name,
                "type": param.type,
                "label": param.label,
                "required": param.required,
                "default": param.default,
                "min": param.minimum,
                "max": param.maximum,
                "options": list(param.options),
            }
            for param in target_spec.params
        ],
    }


__all__ = [
    "BUILTIN_PREFIX",
    "REGISTRY_PREFIX",
    "TargetResolutionError",
    "builtin_name",
    "check_target_available",
    "describe_target",
    "is_builtin_target",
    "load_workflow_catalog",
    "resolve_data_path",
    "resolve_target_component",
]
