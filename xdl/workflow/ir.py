"""WorkflowSnapshot v1 的结构化模型, 解析与结构校验.

本模块只做与节点目录无关的结构检查: 版本, 唯一 ID, 引用完整性, 规模限制, 环检测.
节点类型, 端口类型, target 与参数语义由 :mod:`xdl.workflow.catalog` 和
dashboard 的 compiler 负责, 避免把 XDL 语义塞进纯结构层.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .catalog import WorkflowLimits, load_catalog
from .errors import (
    CYCLE_DETECTED,
    DUPLICATE_EDGE,
    DUPLICATE_ID,
    INVALID_SNAPSHOT,
    LIMIT_EXCEEDED,
    SELF_LOOP,
    UNKNOWN_NODE,
    UNSUPPORTED_VERSION,
    WorkflowValidationError,
    Issue,
    error_issue,
    has_errors,
)

SNAPSHOT_VERSION = 1
SNAPSHOT_KIND = "xdl.workflow.snapshot"
EXECUTION_PROFILES: Tuple[str, ...] = ("schema-v1", "dag-runtime")
DEFAULT_PROFILE = "schema-v1"


@dataclass(frozen=True)
class PortRef:
    """边端点: 节点 ID 加端口名."""

    node: str
    port: str

    def as_dict(self) -> Dict[str, str]:
        return {"node": self.node, "port": self.port}


@dataclass(frozen=True)
class WorkflowNode:
    """一个图节点."""

    id: str
    kind: str
    target: str
    params: Mapping[str, Any] = field(default_factory=dict)
    position: Optional[Mapping[str, float]] = None
    metadata: Optional[Mapping[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "params": dict(self.params),
        }
        if self.position is not None:
            payload["position"] = dict(self.position)
        if self.metadata is not None:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class WorkflowEdge:
    """一条有类型边: 从 source 端口指向 target 端口."""

    id: str
    source: PortRef
    target: PortRef

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from": self.source.as_dict(),
            "to": self.target.as_dict(),
        }


@dataclass(frozen=True)
class StepGraph:
    nodes: Tuple[WorkflowNode, ...]
    edges: Tuple[WorkflowEdge, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                node.as_dict() for node in sorted(self.nodes, key=lambda n: n.id)
            ],
            "edges": [
                edge.as_dict() for edge in sorted(self.edges, key=lambda e: e.id)
            ],
        }


@dataclass(frozen=True)
class WorkflowSnapshot:
    """不可变的 WorkflowSnapshot v1."""

    version: int
    id: str
    name: str
    execution_profile: str
    nodes: Tuple[WorkflowNode, ...]
    edges: Tuple[WorkflowEdge, ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    step_graph: Optional[StepGraph] = None

    def node_map(self) -> Dict[str, WorkflowNode]:
        """返回节点 ID 到节点的映射."""
        return {node.id: node for node in self.nodes}

    def node(self, node_id: str) -> WorkflowNode:
        """按 ID 取节点, 不存在时抛 KeyError."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def nodes_of_kind(self, kind: str) -> Tuple[WorkflowNode, ...]:
        """返回指定类型的全部节点."""
        return tuple(node for node in self.nodes if node.kind == kind)

    def incoming(
        self, node_id: str, port: Optional[str] = None
    ) -> Tuple[WorkflowEdge, ...]:
        """返回指向某节点 (可选限定端口) 的入边."""
        return tuple(
            edge
            for edge in self.edges
            if edge.target.node == node_id
            and (port is None or edge.target.port == port)
        )

    def outgoing(self, node_id: str) -> Tuple[WorkflowEdge, ...]:
        """返回从某节点出发的出边."""
        return tuple(edge for edge in self.edges if edge.source.node == node_id)

    def topological_order(self) -> Tuple[str, ...]:
        """返回稳定拓扑序; 存在环时抛 WorkflowValidationError."""
        return tuple(node_id for node_id, _ in _topological_layers(self))

    def upstream_ids(self, node_id: str) -> Tuple[str, ...]:
        """返回某节点的全部上游节点 ID (不含自身)."""
        seen: set[str] = set()
        stack = [edge.source.node for edge in self.incoming(node_id)]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(edge.source.node for edge in self.incoming(current))
        return tuple(sorted(seen))

    def canonical_payload(self) -> Dict[str, Any]:
        """返回规范化字典: 节点与边按 ID 排序, 便于稳定哈希与比对."""
        payload: Dict[str, Any] = {
            "kind": SNAPSHOT_KIND,
            "version": self.version,
            "name": self.name,
            "execution_profile": self.execution_profile,
            "nodes": [
                node.as_dict() for node in sorted(self.nodes, key=lambda item: item.id)
            ],
            "edges": [
                edge.as_dict() for edge in sorted(self.edges, key=lambda item: item.id)
            ],
            "settings": dict(self.settings),
        }
        if self.id:
            payload["id"] = self.id
        if self.step_graph is not None:
            payload["step_graph"] = self.step_graph.as_dict()
        return payload

    def canonical_json(self) -> str:
        """返回规范化 JSON 文本 (紧凑, 键有序)."""
        return json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def sha256(self) -> str:
        """返回规范化 JSON 的 SHA-256, 用于本进程内的引用与比对."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def validate_structure(
    payload: Any,
    *,
    limits: Optional[WorkflowLimits] = None,
    catalog_path: Optional[str] = None,
) -> List[Issue]:
    """校验快照的结构与规模, 返回全部问题."""
    resolved_limits = limits or load_catalog(catalog_path).limits
    issues: List[Issue] = []
    if not isinstance(payload, Mapping):
        return [error_issue(INVALID_SNAPSHOT, "workflow snapshot 必须是 JSON 对象")]

    version = payload.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        return [
            error_issue(
                UNSUPPORTED_VERSION,
                f"workflow snapshot 必须显式声明整数 version, 收到 {version!r}",
                path="version",
            )
        ]
    if version != SNAPSHOT_VERSION:
        return [
            error_issue(
                UNSUPPORTED_VERSION,
                f"不支持的 workflow snapshot version {version}, 只接受 {SNAPSHOT_VERSION}",
                path="version",
            )
        ]

    issues.extend(_validate_envelope(payload, resolved_limits))
    issues.extend(_validate_size(payload, resolved_limits))
    nodes, node_issues = _validate_nodes(payload.get("nodes"), resolved_limits)
    issues.extend(node_issues)
    node_ids = {node.id for node in nodes}
    edges, edge_issues = _validate_edges(
        payload.get("edges"), node_ids, resolved_limits
    )
    issues.extend(edge_issues)
    issues.extend(_validate_cycles(nodes, edges))
    if "step_graph" in payload:
        _, step_issues = _parse_step_graph(payload["step_graph"], resolved_limits)
        issues.extend(step_issues)
    return issues


def _parse_step_graph(
    raw: Any, limits: WorkflowLimits
) -> Tuple[Optional[StepGraph], List[Issue]]:
    if not isinstance(raw, Mapping) or set(raw) != {"nodes", "edges"}:
        return None, [
            error_issue(
                INVALID_SNAPSHOT,
                "step_graph 必须包含 nodes 与 edges",
                path="step_graph",
            )
        ]
    nodes, issues = _validate_nodes(raw["nodes"], limits)
    edges, edge_issues = _validate_edges(
        raw["edges"], {node.id for node in nodes}, limits
    )
    issues.extend(edge_issues)
    issues.extend(_validate_cycles(nodes, edges))
    prefixed = [
        replace(issue, path=f"step_graph/{issue.path}" if issue.path else "step_graph")
        for issue in issues
    ]
    if has_errors(issues):
        return None, prefixed
    return StepGraph(tuple(nodes), tuple(edges)), prefixed


def parse_snapshot(
    payload: Any, *, catalog_path: Optional[str] = None
) -> WorkflowSnapshot:
    """解析并返回不可变快照; 结构不合法时抛 WorkflowValidationError."""
    issues = validate_structure(payload, catalog_path=catalog_path)
    if has_errors(issues):
        raise WorkflowValidationError(issues)
    return _build_snapshot(payload, load_catalog(catalog_path).limits)


def load_snapshot_file(path: str | Any) -> Tuple[WorkflowSnapshot, str]:
    """读取快照文件并返回 (快照, 文件字节的 SHA-256).

    哈希基于文件原始字节, 与 Go dashboard 写入的 canonical JSON 完全一致.
    """
    from pathlib import Path

    resolved = Path(path).expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    return parse_snapshot(payload), digest


def hash_snapshot_bytes(raw: bytes) -> str:
    """返回快照原始字节的 SHA-256."""
    return hashlib.sha256(raw).hexdigest()


def _build_snapshot(
    payload: Mapping[str, Any], limits: WorkflowLimits
) -> WorkflowSnapshot:
    nodes = tuple(
        WorkflowNode(
            id=str(raw["id"]),
            kind=str(raw["kind"]),
            target=str(raw["target"]),
            params=dict(raw.get("params") or {}),
            position=_position_of(raw.get("position")),
            metadata=dict(raw["metadata"])
            if isinstance(raw.get("metadata"), Mapping)
            else None,
        )
        for raw in payload.get("nodes") or []
    )
    edges = tuple(
        WorkflowEdge(
            id=_edge_id(raw, index),
            source=PortRef(str(raw["from"]["node"]), str(raw["from"]["port"])),
            target=PortRef(str(raw["to"]["node"]), str(raw["to"]["port"])),
        )
        for index, raw in enumerate(payload.get("edges") or [])
    )
    settings = payload.get("settings")
    step_graph = None
    if "step_graph" in payload:
        step_graph, _ = _parse_step_graph(payload["step_graph"], limits)
    return WorkflowSnapshot(
        version=int(payload["version"]),
        id=str(payload.get("id") or ""),
        name=str(payload.get("name") or ""),
        execution_profile=str(payload.get("execution_profile") or DEFAULT_PROFILE),
        nodes=nodes,
        edges=edges,
        settings=dict(settings) if isinstance(settings, Mapping) else {},
        step_graph=step_graph,
    )


def _edge_id(raw: Mapping[str, Any], index: int) -> str:
    """返回边 ID; 缺省时与结构校验一致地按索引生成."""
    value = raw.get("id")
    if isinstance(value, str) and value.strip():
        return value
    return f"edge-{index + 1}"


def _position_of(value: Any) -> Optional[Mapping[str, float]]:
    if not isinstance(value, Mapping):
        return None
    try:
        return {"x": float(value["x"]), "y": float(value["y"])}
    except (KeyError, TypeError, ValueError):
        return None


def _validate_envelope(
    payload: Mapping[str, Any], limits: WorkflowLimits
) -> List[Issue]:
    issues: List[Issue] = []
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        issues.append(
            error_issue(INVALID_SNAPSHOT, "name 必须是非空字符串", path="name")
        )
    elif len(name) > limits.max_name_length:
        issues.append(
            error_issue(
                LIMIT_EXCEEDED,
                f"name 长度 {len(name)} 超过上限 {limits.max_name_length}",
                path="name",
            )
        )
    snapshot_id = payload.get("id")
    if snapshot_id is not None:
        if not isinstance(snapshot_id, str):
            issues.append(error_issue(INVALID_SNAPSHOT, "id 必须是字符串", path="id"))
        elif len(snapshot_id) > limits.max_id_length:
            issues.append(
                error_issue(
                    LIMIT_EXCEEDED,
                    f"id 长度超过上限 {limits.max_id_length}",
                    path="id",
                )
            )
    profile = payload.get("execution_profile")
    if not isinstance(profile, str) or not profile.strip():
        issues.append(
            error_issue(
                INVALID_SNAPSHOT,
                "execution_profile 必须显式声明, 不接受默认猜测",
                path="execution_profile",
            )
        )
    settings = payload.get("settings")
    if settings is not None and not isinstance(settings, Mapping):
        issues.append(
            error_issue(INVALID_SNAPSHOT, "settings 必须是 JSON 对象", path="settings")
        )
    return issues


def _validate_size(payload: Mapping[str, Any], limits: WorkflowLimits) -> List[Issue]:
    issues: List[Issue] = []
    try:
        encoded = json.dumps(payload, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        return [error_issue(INVALID_SNAPSHOT, f"快照无法序列化为 JSON: {exc}")]
    if len(encoded) > limits.max_snapshot_bytes:
        issues.append(
            error_issue(
                LIMIT_EXCEEDED,
                f"快照大小 {len(encoded)} 字节超过上限 {limits.max_snapshot_bytes}",
            )
        )
    return issues


def _validate_nodes(
    raw_nodes: Any, limits: WorkflowLimits
) -> Tuple[List[WorkflowNode], List[Issue]]:
    issues: List[Issue] = []
    if not isinstance(raw_nodes, list):
        return [], [error_issue(INVALID_SNAPSHOT, "nodes 必须是数组", path="nodes")]
    if not raw_nodes:
        issues.append(error_issue(INVALID_SNAPSHOT, "nodes 不能为空", path="nodes"))
    if len(raw_nodes) > limits.max_nodes:
        issues.append(
            error_issue(
                LIMIT_EXCEEDED,
                f"节点数 {len(raw_nodes)} 超过上限 {limits.max_nodes}",
                path="nodes",
            )
        )
    nodes: List[WorkflowNode] = []
    seen: Dict[str, int] = {}
    for index, raw in enumerate(raw_nodes):
        path = f"nodes/{index}"
        if not isinstance(raw, Mapping):
            issues.append(
                error_issue(INVALID_SNAPSHOT, "节点必须是 JSON 对象", path=path)
            )
            continue
        node_id = raw.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            issues.append(
                error_issue(
                    INVALID_SNAPSHOT, "节点 id 必须是非空字符串", path=f"{path}/id"
                )
            )
            continue
        if len(node_id) > limits.max_id_length:
            issues.append(
                error_issue(
                    LIMIT_EXCEEDED,
                    f"节点 id 长度超过上限 {limits.max_id_length}",
                    path=f"{path}/id",
                    node_id=node_id,
                )
            )
        if node_id in seen:
            issues.append(
                error_issue(
                    DUPLICATE_ID,
                    f"节点 id {node_id!r} 重复 (首次出现在 nodes/{seen[node_id]})",
                    path=f"{path}/id",
                    node_id=node_id,
                )
            )
            continue
        seen[node_id] = index
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            issues.append(
                error_issue(
                    INVALID_SNAPSHOT,
                    "节点 kind 必须是非空字符串",
                    path=f"{path}/kind",
                    node_id=node_id,
                )
            )
            continue
        target = raw.get("target")
        if not isinstance(target, str) or not target.strip():
            issues.append(
                error_issue(
                    INVALID_SNAPSHOT,
                    "节点 target 必须是非空字符串",
                    path=f"{path}/target",
                    node_id=node_id,
                )
            )
            continue
        params = raw.get("params")
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            issues.append(
                error_issue(
                    INVALID_SNAPSHOT,
                    "节点 params 必须是 JSON 对象",
                    path=f"{path}/params",
                    node_id=node_id,
                )
            )
            continue
        issues.extend(
            _validate_params_shape(
                params, limits, path=f"{path}/params", node_id=node_id
            )
        )
        nodes.append(
            WorkflowNode(
                id=node_id,
                kind=kind,
                target=target,
                params=dict(params),
                position=_position_of(raw.get("position")),
                metadata=dict(raw["metadata"])
                if isinstance(raw.get("metadata"), Mapping)
                else None,
            )
        )
    return nodes, issues


def _validate_params_shape(
    params: Mapping[str, Any], limits: WorkflowLimits, *, path: str, node_id: str
) -> List[Issue]:
    issues: List[Issue] = []
    try:
        encoded = json.dumps(params, separators=(",", ":"), default=str)
    except (TypeError, ValueError) as exc:
        return [
            error_issue(
                INVALID_SNAPSHOT,
                f"params 无法序列化为 JSON: {exc}",
                path=path,
                node_id=node_id,
            )
        ]
    if len(encoded.encode("utf-8")) > limits.max_params_bytes:
        issues.append(
            error_issue(
                LIMIT_EXCEEDED,
                f"params 大小超过上限 {limits.max_params_bytes} 字节",
                path=path,
                node_id=node_id,
            )
        )
    depth = _max_depth(params)
    if depth > limits.max_params_depth:
        issues.append(
            error_issue(
                LIMIT_EXCEEDED,
                f"params 嵌套深度 {depth} 超过上限 {limits.max_params_depth}",
                path=path,
                node_id=node_id,
            )
        )
    return issues


def _max_depth(value: Any, current: int = 0) -> int:
    if isinstance(value, Mapping):
        if not value:
            return current
        return max(_max_depth(item, current + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        if not value:
            return current
        return max(_max_depth(item, current + 1) for item in value)
    return current


def _validate_edges(
    raw_edges: Any, node_ids: set[str], limits: WorkflowLimits
) -> Tuple[List[WorkflowEdge], List[Issue]]:
    issues: List[Issue] = []
    if raw_edges is None:
        raw_edges = []
    if not isinstance(raw_edges, list):
        return [], [error_issue(INVALID_SNAPSHOT, "edges 必须是数组", path="edges")]
    if len(raw_edges) > limits.max_edges:
        issues.append(
            error_issue(
                LIMIT_EXCEEDED,
                f"边数 {len(raw_edges)} 超过上限 {limits.max_edges}",
                path="edges",
            )
        )
    edges: List[WorkflowEdge] = []
    seen_ids: Dict[str, int] = {}
    seen_pairs: Dict[Tuple[str, str, str, str], str] = {}
    for index, raw in enumerate(raw_edges):
        path = f"edges/{index}"
        if not isinstance(raw, Mapping):
            issues.append(
                error_issue(INVALID_SNAPSHOT, "边必须是 JSON 对象", path=path)
            )
            continue
        edge_id = raw.get("id")
        if edge_id is None:
            edge_id = f"edge-{index + 1}"
        if not isinstance(edge_id, str) or not edge_id.strip():
            issues.append(
                error_issue(
                    INVALID_SNAPSHOT, "边 id 必须是非空字符串", path=f"{path}/id"
                )
            )
            continue
        if edge_id in seen_ids:
            issues.append(
                error_issue(
                    DUPLICATE_ID,
                    f"边 id {edge_id!r} 重复 (首次出现在 edges/{seen_ids[edge_id]})",
                    path=f"{path}/id",
                    edge_id=edge_id,
                )
            )
            continue
        seen_ids[edge_id] = index
        source = _parse_port_ref(raw.get("from"), path=f"{path}/from", edge_id=edge_id)
        target = _parse_port_ref(raw.get("to"), path=f"{path}/to", edge_id=edge_id)
        if source is None or target is None:
            issues.extend(_port_ref_issues(raw, path, edge_id))
            continue
        missing = False
        for ref, side in ((source, "from"), (target, "to")):
            if ref.node not in node_ids:
                issues.append(
                    error_issue(
                        UNKNOWN_NODE,
                        f"边 {edge_id} 的 {side} 引用不存在的节点 {ref.node!r}",
                        path=f"{path}/{side}/node",
                        edge_id=edge_id,
                    )
                )
                missing = True
        if missing:
            continue
        if source.node == target.node:
            issues.append(
                error_issue(
                    SELF_LOOP,
                    f"边 {edge_id} 连接了节点自身 {source.node!r}",
                    path=path,
                    node_id=source.node,
                    edge_id=edge_id,
                )
            )
            continue
        pair = (source.node, source.port, target.node, target.port)
        if pair in seen_pairs:
            issues.append(
                error_issue(
                    DUPLICATE_EDGE,
                    f"边 {edge_id} 与 {seen_pairs[pair]} 重复连接 "
                    f"{source.node}.{source.port} -> {target.node}.{target.port}",
                    path=path,
                    edge_id=edge_id,
                )
            )
            continue
        seen_pairs[pair] = edge_id
        edges.append(WorkflowEdge(id=edge_id, source=source, target=target))
    return edges, issues


def _parse_port_ref(value: Any, *, path: str, edge_id: str) -> Optional[PortRef]:
    if not isinstance(value, Mapping):
        return None
    node = value.get("node")
    port = value.get("port")
    if not isinstance(node, str) or not node.strip():
        return None
    if not isinstance(port, str) or not port.strip():
        return None
    return PortRef(node=node, port=port)


def _port_ref_issues(
    raw: Mapping[str, Any], path: str, edge_id: str
) -> Iterator[Issue]:
    for side in ("from", "to"):
        value = raw.get(side)
        if not isinstance(value, Mapping):
            yield error_issue(
                INVALID_SNAPSHOT,
                f"边 {edge_id} 缺少 {side} 端点对象",
                path=f"{path}/{side}",
                edge_id=edge_id,
            )
            continue
        for key in ("node", "port"):
            if not isinstance(value.get(key), str) or not str(value.get(key)).strip():
                yield error_issue(
                    INVALID_SNAPSHOT,
                    f"边 {edge_id} 的 {side}.{key} 必须是非空字符串",
                    path=f"{path}/{side}/{key}",
                    edge_id=edge_id,
                )


def _validate_cycles(
    nodes: Sequence[WorkflowNode], edges: Sequence[WorkflowEdge]
) -> List[Issue]:
    snapshot = WorkflowSnapshot(
        version=SNAPSHOT_VERSION,
        id="",
        name="",
        execution_profile=DEFAULT_PROFILE,
        nodes=tuple(nodes),
        edges=tuple(edges),
        settings={},
    )
    remaining = _topological_layers(snapshot, detect_cycles=True)
    leftover = [node_id for node_id, layer in remaining if layer is None]
    if not leftover:
        return []
    return [
        error_issue(
            CYCLE_DETECTED,
            "节点图存在环: " + " -> ".join(sorted(leftover)),
            node_id=sorted(leftover)[0],
        )
    ]


def _topological_layers(
    snapshot: WorkflowSnapshot, *, detect_cycles: bool = False
) -> List[Tuple[str, Optional[int]]]:
    """按 Kahn 算法返回 (node_id, layer) 列表; 有环时剩余节点 layer 为 None."""
    indegree: Dict[str, int] = {node.id: 0 for node in snapshot.nodes}
    outgoing: Dict[str, List[str]] = {node.id: [] for node in snapshot.nodes}
    for edge in snapshot.edges:
        if edge.target.node not in indegree or edge.source.node not in outgoing:
            continue
        indegree[edge.target.node] += 1
        outgoing[edge.source.node].append(edge.target.node)
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    layers: List[Tuple[str, Optional[int]]] = []
    layer = 0
    while ready:
        next_ready: List[str] = []
        for node_id in ready:
            layers.append((node_id, layer))
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    next_ready.append(target)
        ready = sorted(next_ready)
        layer += 1
    if detect_cycles:
        visited = {node_id for node_id, _ in layers}
        for node_id in sorted(indegree):
            if node_id not in visited:
                layers.append((node_id, None))
    elif len(layers) != len(snapshot.nodes):
        raise WorkflowValidationError(
            [
                error_issue(
                    CYCLE_DETECTED,
                    "节点图存在环, 无法计算拓扑序",
                )
            ]
        )
    return layers


__all__ = [
    "DEFAULT_PROFILE",
    "EXECUTION_PROFILES",
    "SNAPSHOT_KIND",
    "SNAPSHOT_VERSION",
    "PortRef",
    "WorkflowEdge",
    "WorkflowNode",
    "WorkflowSnapshot",
    "hash_snapshot_bytes",
    "load_snapshot_file",
    "parse_snapshot",
    "validate_structure",
]
