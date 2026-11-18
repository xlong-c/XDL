"""节点目录: 有类型 DAG 的节点, 端口, 参数与 target 契约.

目录的唯一实体来源是包内的 ``node_catalog.json``. 前端通过 dashboard API 读取同一
份 JSON, Go 只转发字节并读取其中的 ``limits``, 不维护第二份节点语义.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .errors import (
    INVALID_PARAM,
    PATH_NOT_ALLOWED,
    UNKNOWN_NODE_KIND,
    UNKNOWN_TARGET,
    Issue,
    error_issue,
)

if TYPE_CHECKING:
    from .ir import WorkflowSnapshot

CATALOG_FILENAME = "node_catalog.json"

PARAM_TYPES = frozenset(
    {
        "integer",
        "number",
        "boolean",
        "string",
        "enum",
        "list_integer",
        "list_number",
        "data_path",
    }
)


@dataclass(frozen=True)
class PortSpec:
    """一个输入或输出端口的声明.

    ``multi`` 为真表示该输入端口允许同时接入多个上游 (例如多个 loss 节点按
    weight 加权), 其余端口只允许一个来源.
    """

    name: str
    type: str
    required: bool
    multi: bool = False


@dataclass(frozen=True)
class ParamSpec:
    """一个组件参数的声明."""

    name: str
    type: str
    label: str
    required: bool
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    options: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetSpec:
    """一个节点可选的组件 target 及其参数声明."""

    target: str
    label: str
    summary: str
    params: Tuple[ParamSpec, ...]


@dataclass(frozen=True)
class NodeKindSpec:
    """一种节点类型的完整声明."""

    kind: str
    label: str
    group: str
    summary: str
    profiles: Tuple[str, ...]
    inputs: Tuple[PortSpec, ...]
    outputs: Tuple[PortSpec, ...]
    targets: Tuple[TargetSpec, ...]


@dataclass(frozen=True)
class WorkflowLimits:
    """快照规模限制, Go dashboard 与 Python compiler 共用同一组数字."""

    max_nodes: int
    max_edges: int
    max_params_depth: int
    max_params_bytes: int
    max_snapshot_bytes: int
    max_name_length: int
    max_id_length: int


@dataclass(frozen=True)
class NodeCatalog:
    """完整节点目录."""

    version: int
    catalog_version: str
    limits: WorkflowLimits
    profiles: Mapping[str, str]
    port_types: Mapping[str, str]
    kinds: Mapping[str, NodeKindSpec]
    step_kinds: Mapping[str, NodeKindSpec]

    def kind(self, name: str) -> Optional[NodeKindSpec]:
        """按名称查找节点类型."""
        return self.kinds.get(name)

    def kind_names(self) -> Tuple[str, ...]:
        """返回全部受支持的节点类型名称."""
        return tuple(self.kinds)

    def target(self, kind: str, target: str) -> Optional[TargetSpec]:
        """按节点类型和 target 查找 target 声明."""
        spec = self.kinds.get(kind)
        if spec is None:
            return None
        for candidate in spec.targets:
            if candidate.target == target:
                return candidate
        return None

    def profiles_of(self, kind: str) -> Tuple[str, ...]:
        """返回节点类型支持的执行 profile."""
        spec = self.kinds.get(kind)
        return () if spec is None else spec.profiles


def default_catalog_path() -> Path:
    """返回包内节点目录 JSON 的路径."""
    return Path(__file__).with_name(CATALOG_FILENAME)


@lru_cache(maxsize=4)
def _load_catalog_cached(path_text: str) -> NodeCatalog:
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    return parse_catalog(payload)


def load_catalog(path: Optional[str | Path] = None) -> NodeCatalog:
    """加载节点目录, 默认使用包内 JSON."""
    resolved = default_catalog_path() if path is None else Path(path)
    return _load_catalog_cached(str(resolved.expanduser().resolve()))


def parse_catalog(payload: Mapping[str, Any]) -> NodeCatalog:
    """把节点目录 JSON 解析为结构化对象."""
    version = int(payload.get("version", 0))
    if version != 1:
        raise ValueError(f"unsupported node catalog version: {version}")
    raw_limits = payload.get("limits") or {}
    limits = WorkflowLimits(
        max_nodes=int(raw_limits.get("max_nodes", 64)),
        max_edges=int(raw_limits.get("max_edges", 256)),
        max_params_depth=int(raw_limits.get("max_params_depth", 6)),
        max_params_bytes=int(raw_limits.get("max_params_bytes", 65536)),
        max_snapshot_bytes=int(raw_limits.get("max_snapshot_bytes", 262144)),
        max_name_length=int(raw_limits.get("max_name_length", 120)),
        max_id_length=int(raw_limits.get("max_id_length", 64)),
    )
    kinds: Dict[str, NodeKindSpec] = {}
    for raw_kind in payload.get("kinds") or []:
        spec = _parse_kind(raw_kind)
        if spec.kind in kinds:
            raise ValueError(f"duplicate node kind in catalog: {spec.kind}")
        kinds[spec.kind] = spec
    step_kinds = {
        spec.kind: spec for spec in map(_parse_kind, payload.get("step_kinds") or [])
    }
    return NodeCatalog(
        version=version,
        catalog_version=str(payload.get("catalog_version", "1")),
        limits=limits,
        profiles=dict(payload.get("profiles") or {}),
        port_types=dict(payload.get("port_types") or {}),
        kinds=kinds,
        step_kinds=step_kinds,
    )


def assembly_port_types(
    snapshot: WorkflowSnapshot, catalog: NodeCatalog
) -> Dict[Tuple[str, str], str]:
    """Resolve passthrough ports from upstream declarations in topological order."""
    nodes = snapshot.node_map()
    types: Dict[Tuple[str, str], str] = {}
    for node_id in snapshot.topological_order():
        node = nodes[node_id]
        spec = catalog.kind(node.kind)
        if spec is None:
            continue
        for port in spec.outputs:
            types[node_id, port.name] = port.type
        if node.kind in {"identity", "compose"}:
            inputs = {
                edge.target.port: types.get((edge.source.node, edge.source.port))
                for edge in snapshot.incoming(node_id)
            }
            passed = inputs.get("in")
            if node.kind == "identity" and passed is not None:
                types[node_id, "out"] = passed
            left, right = inputs.get("left"), inputs.get("right")
            if (
                node.kind == "compose"
                and left is not None
                and left == right
                and left in {"transform", "loss"}
            ):
                types[node_id, "out"] = left
    return types


def _parse_kind(payload: Mapping[str, Any]) -> NodeKindSpec:
    kind = str(payload.get("kind", ""))
    if not kind:
        raise ValueError("node catalog kind requires a name")
    targets: List[TargetSpec] = []
    for raw_target in payload.get("targets") or []:
        targets.append(
            TargetSpec(
                target=str(raw_target.get("target", "")),
                label=str(raw_target.get("label", "")),
                summary=str(raw_target.get("summary", "")),
                params=tuple(
                    _parse_param(item) for item in raw_target.get("params") or []
                ),
            )
        )
    return NodeKindSpec(
        kind=kind,
        label=str(payload.get("label", kind)),
        group=str(payload.get("group", "")),
        summary=str(payload.get("summary", "")),
        profiles=tuple(str(item) for item in payload.get("profiles") or []),
        inputs=tuple(_parse_port(item) for item in payload.get("inputs") or []),
        outputs=tuple(_parse_port(item) for item in payload.get("outputs") or []),
        targets=tuple(targets),
    )


def _parse_port(payload: Mapping[str, Any]) -> PortSpec:
    return PortSpec(
        name=str(payload.get("name", "")),
        type=str(payload.get("type", "")),
        required=bool(payload.get("required", False)),
        multi=bool(payload.get("multi", False)),
    )


def _parse_param(payload: Mapping[str, Any]) -> ParamSpec:
    param_type = str(payload.get("type", ""))
    if param_type not in PARAM_TYPES:
        raise ValueError(f"unsupported catalog param type: {param_type}")
    minimum = payload.get("min")
    maximum = payload.get("max")
    return ParamSpec(
        name=str(payload.get("name", "")),
        type=param_type,
        label=str(payload.get("label", payload.get("name", ""))),
        required=bool(payload.get("required", False)),
        default=payload.get("default"),
        minimum=None if minimum is None else float(minimum),
        maximum=None if maximum is None else float(maximum),
        options=tuple(str(item) for item in payload.get("options") or []),
    )


def check_kind_and_target(
    catalog: NodeCatalog,
    *,
    node_id: str,
    kind: str,
    target: str,
    path: str = "",
) -> Tuple[Optional[NodeKindSpec], Optional[TargetSpec], List[Issue]]:
    """校验节点类型和 target 是否在目录内."""
    issues: List[Issue] = []
    kind_spec = catalog.kind(kind)
    if kind_spec is None:
        issues.append(
            error_issue(
                UNKNOWN_NODE_KIND,
                f"未知节点类型 {kind!r}; 受支持类型: "
                + ", ".join(sorted(catalog.kind_names())),
                path=path or f"nodes/{node_id}/kind",
                node_id=node_id,
            )
        )
        return None, None, issues
    target_spec = catalog.target(kind, target)
    if target_spec is None:
        allowed = ", ".join(item.target for item in kind_spec.targets)
        issues.append(
            error_issue(
                UNKNOWN_TARGET,
                f"节点 {node_id} 的 target {target!r} 不在 {kind} 目录内; "
                f"允许: {allowed}",
                path=path or f"nodes/{node_id}/target",
                node_id=node_id,
            )
        )
        return kind_spec, None, issues
    return kind_spec, target_spec, issues


def validate_params(
    target_spec: TargetSpec,
    params: Mapping[str, Any],
    *,
    node_id: str,
    path_prefix: str = "",
) -> List[Issue]:
    """按 target 参数声明校验节点参数."""
    issues: List[Issue] = []
    declared = {spec.name: spec for spec in target_spec.params}
    for name in params:
        if name not in declared:
            issues.append(
                error_issue(
                    INVALID_PARAM,
                    f"{target_spec.target} 不接受参数 {name!r}; 允许: "
                    + (", ".join(sorted(declared)) or "(无)"),
                    path=f"{path_prefix}/{name}".lstrip("/"),
                    node_id=node_id,
                )
            )
    for spec in target_spec.params:
        value = params.get(spec.name)
        if value is None:
            if spec.required:
                issues.append(
                    error_issue(
                        INVALID_PARAM,
                        f"{target_spec.target} 缺少必需参数 {spec.name!r}",
                        path=f"{path_prefix}/{spec.name}".lstrip("/"),
                        node_id=node_id,
                    )
                )
            continue
        issues.extend(
            _validate_param_value(
                spec,
                value,
                node_id=node_id,
                path=f"{path_prefix}/{spec.name}".lstrip("/"),
            )
        )
    return issues


def _validate_param_value(
    spec: ParamSpec,
    value: Any,
    *,
    node_id: str,
    path: str,
) -> List[Issue]:
    def fail(message: str) -> List[Issue]:
        return [
            error_issue(
                INVALID_PARAM,
                f"参数 {spec.name}: {message}",
                path=path,
                node_id=node_id,
            )
        ]

    if spec.type == "integer":
        if not _is_integer(value):
            return fail("必须是整数")
        return _check_range(spec, float(value), node_id=node_id, path=path)
    if spec.type == "number":
        if not _is_number(value):
            return fail("必须是有限数字")
        return _check_range(spec, float(value), node_id=node_id, path=path)
    if spec.type == "boolean":
        return [] if isinstance(value, bool) else fail("必须是布尔值")
    if spec.type == "string":
        return [] if isinstance(value, str) else fail("必须是字符串")
    if spec.type == "enum":
        if not isinstance(value, str) or value not in spec.options:
            return fail("必须是 " + ", ".join(spec.options) + " 之一")
        return []
    if spec.type == "list_integer":
        if not isinstance(value, (list, tuple)) or not all(
            _is_integer(item) for item in value
        ):
            return fail("必须是整数数组")
        return []
    if spec.type == "list_number":
        if not isinstance(value, (list, tuple)) or not all(
            _is_number(item) for item in value
        ):
            return fail("必须是数字数组")
        return []
    if spec.type == "data_path":
        if not isinstance(value, str) or not value.strip():
            return fail("必须是相对数据路径")
        return _check_relative_path(value, node_id=node_id, path=path)
    return fail(f"未知参数类型 {spec.type}")


def _check_range(
    spec: ParamSpec, value: float, *, node_id: str, path: str
) -> List[Issue]:
    issues: List[Issue] = []
    if spec.minimum is not None and value < spec.minimum:
        issues.append(
            error_issue(
                INVALID_PARAM,
                f"参数 {spec.name}: 必须 >= {spec.minimum}",
                path=path,
                node_id=node_id,
            )
        )
    if spec.maximum is not None and value > spec.maximum:
        issues.append(
            error_issue(
                INVALID_PARAM,
                f"参数 {spec.name}: 必须 <= {spec.maximum}",
                path=path,
                node_id=node_id,
            )
        )
    return issues


def _check_relative_path(value: str, *, node_id: str, path: str) -> List[Issue]:
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        return [
            error_issue(
                PATH_NOT_ALLOWED,
                f"数据路径必须是相对路径, 不允许 {value!r}",
                path=path,
                node_id=node_id,
            )
        ]
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        return [
            error_issue(
                PATH_NOT_ALLOWED,
                f"数据路径必须位于 data 根目录内, 不允许 {value!r}",
                path=path,
                node_id=node_id,
            )
        ]
    return []


def apply_param_defaults(
    target_spec: TargetSpec, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """补齐 target 参数声明中的默认值, 不覆盖显式值."""
    resolved = dict(params)
    for spec in target_spec.params:
        if spec.name not in resolved and spec.default is not None:
            resolved[spec.name] = spec.default
    return resolved


def _is_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and value.is_integer()


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == value and value not in (float("inf"), float("-inf"))
    return False


def iter_kind_targets(catalog: NodeCatalog) -> Iterable[Tuple[str, str]]:
    """遍历目录中的 (kind, target) 组合."""
    for kind_spec in catalog.kinds.values():
        for target_spec in kind_spec.targets:
            yield kind_spec.kind, target_spec.target


__all__ = [
    "CATALOG_FILENAME",
    "PARAM_TYPES",
    "NodeCatalog",
    "NodeKindSpec",
    "ParamSpec",
    "PortSpec",
    "TargetSpec",
    "WorkflowLimits",
    "apply_param_defaults",
    "check_kind_and_target",
    "default_catalog_path",
    "iter_kind_targets",
    "load_catalog",
    "parse_catalog",
    "validate_params",
]
