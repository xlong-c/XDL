"""工作流语义校验.

职责:
- 节点类型, target 与参数是否命中节点目录.
- 节点与图是否支持声明的执行 profile.
- 边的端口是否存在, 端口类型是否兼容, 输入端口是否只有一个来源.
- 每个 profile 必需的节点角色与必需输入端口.
- 图连通性: 每个节点都必须在 trainer 所在的连通分量里.
- dag-runtime 的输出必须被消费 (sink 或 trainer.run 除外).
- 可选地校验 target 在当前 XDL 环境中真实存在.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from xdl.workflow import (
    Issue,
    NodeCatalog,
    WorkflowSnapshot,
    has_errors,
    load_catalog,
    parse_snapshot,
    validate_structure,
)
from xdl.workflow.catalog import (
    NodeKindSpec,
    TargetSpec,
    apply_param_defaults,
    assembly_port_types,
    check_kind_and_target,
    validate_params,
)
from xdl.workflow.errors import (
    DANGLING_OUTPUT,
    DUPLICATE_ROLE,
    INVALID_PARAM,
    MISSING_PORT,
    MISSING_REQUIRED_NODE,
    PORT_CONFLICT,
    TYPE_MISMATCH,
    UNKNOWN_PORT,
    UNREACHABLE_NODE,
    UNSUPPORTED_PROFILE,
    error_issue,
    warning_issue,
)
from xdl.workflow.ir import WorkflowEdge, WorkflowNode
from xdl.workflow.errors import WorkflowValidationError
from xdl.workflow.step_graph import compile_step_graph

from .workflow_catalog import check_target_available

PROFILE_SCHEMA_V1 = "schema-v1"
PROFILE_DAG_RUNTIME = "dag-runtime"

# 每个 profile 必须唯一存在的节点角色.
UNIQUE_ROLES: Tuple[str, ...] = ("trainer", "model", "optimizer")
# 每个 profile 至少存在一个的节点角色.
REQUIRED_ROLES: Tuple[str, ...] = ("trainer", "model", "optimizer", "loss")
# 允许最多一个的节点角色.
OPTIONAL_UNIQUE_ROLES: Tuple[str, ...] = ("scheduler", "checkpoint")


@dataclass(frozen=True)
class ResolvedNode:
    """校验通过后的节点: 目录声明 + 生效参数."""

    node: WorkflowNode
    kind: NodeKindSpec
    target: TargetSpec
    params: Mapping[str, object]

    @property
    def id(self) -> str:
        """节点 ID."""
        return self.node.id

    def param(self, name: str, default: object = None) -> object:
        """读取生效参数."""
        return self.params.get(name, default)


@dataclass
class WorkflowReport:
    """一次语义校验的完整结果."""

    execution_profile: str
    snapshot: Optional[WorkflowSnapshot] = None
    nodes: Dict[str, ResolvedNode] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    warnings: List[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """没有 error 级问题时为真."""
        return not has_errors(self.issues)

    def node(self, node_id: str) -> ResolvedNode:
        """按 ID 取已解析节点."""
        return self.nodes[node_id]

    def of_kind(self, kind: str) -> Tuple[ResolvedNode, ...]:
        """返回某类型的全部已解析节点 (按节点 ID 排序)."""
        return tuple(
            self.nodes[node_id]
            for node_id in sorted(self.nodes)
            if self.nodes[node_id].kind.kind == kind
        )

    def source_of(self, node_id: str, port: str) -> Optional[ResolvedNode]:
        """返回某输入端口唯一来源节点."""
        if self.snapshot is None:
            return None
        edges = self.snapshot.incoming(node_id, port)
        if not edges:
            return None
        return self.nodes.get(edges[0].source.node)

    def input_nodes(self, node_id: str, port: str) -> Tuple[ResolvedNode, ...]:
        """返回某输入端口连接的全部来源节点."""
        if self.snapshot is None:
            return ()
        return tuple(
            self.nodes[edge.source.node]
            for edge in self.snapshot.incoming(node_id, port)
            if edge.source.node in self.nodes
        )

    def add(self, issue: Issue) -> None:
        """追加一条问题, 按 level 分流."""
        if issue.level == "error":
            self.issues.append(issue)
        else:
            self.warnings.append(issue)


def validate_workflow(
    payload: object,
    *,
    catalog: Optional[NodeCatalog] = None,
    data_root: Optional[str | Path] = None,
    check_registry: bool = True,
) -> WorkflowReport:
    """校验一份快照, 返回全部问题与已解析节点."""
    resolved_catalog = catalog or load_catalog()
    structure_issues = validate_structure(payload, limits=resolved_catalog.limits)
    profile = ""
    if isinstance(payload, Mapping):
        raw_profile = payload.get("execution_profile")
        profile = raw_profile if isinstance(raw_profile, str) else ""
    report = WorkflowReport(execution_profile=profile)
    for issue in structure_issues:
        report.add(issue)
    if has_errors(structure_issues):
        return report

    snapshot = parse_snapshot(payload)
    report.snapshot = snapshot
    report.execution_profile = snapshot.execution_profile

    _resolve_nodes(report, snapshot, resolved_catalog, check_registry)
    port_types = assembly_port_types(snapshot, resolved_catalog)
    _validate_edges(report, snapshot, port_types)
    _validate_bindings(report, snapshot, port_types)
    _validate_required_inputs(report, snapshot)
    _validate_roles(report, snapshot)
    _validate_connectivity(report, snapshot)
    if snapshot.execution_profile == PROFILE_DAG_RUNTIME:
        _validate_outputs_consumed(report, snapshot)
    if snapshot.step_graph is not None:
        try:
            compile_step_graph(snapshot, resolved_catalog)
        except WorkflowValidationError as exc:
            for issue in exc.issues:
                report.add(issue)
    return report


def _resolve_nodes(
    report: WorkflowReport,
    snapshot: WorkflowSnapshot,
    catalog: NodeCatalog,
    check_registry: bool,
) -> None:
    for node in snapshot.nodes:
        kind_spec, target_spec, issues = check_kind_and_target(
            catalog,
            node_id=node.id,
            kind=node.kind,
            target=node.target,
            path=f"nodes/{node.id}/target",
        )
        for issue in issues:
            report.add(issue)
        if kind_spec is None or target_spec is None:
            continue
        if snapshot.execution_profile not in kind_spec.profiles:
            report.add(
                error_issue(
                    UNSUPPORTED_PROFILE,
                    f"节点 {node.id} 的类型 {node.kind} 不支持执行 profile "
                    f"{snapshot.execution_profile!r}; 支持: "
                    + ", ".join(kind_spec.profiles),
                    path=f"nodes/{node.id}/kind",
                    node_id=node.id,
                )
            )
            continue
        for issue in validate_params(
            target_spec,
            node.params,
            node_id=node.id,
            path_prefix=f"nodes/{node.id}/params",
        ):
            report.add(issue)
        if check_registry:
            issue = check_target_available(
                node.id, node.kind, node.target, path=f"nodes/{node.id}/target"
            )
            if issue is not None:
                report.add(issue)
        report.nodes[node.id] = ResolvedNode(
            node=node,
            kind=kind_spec,
            target=target_spec,
            params=apply_param_defaults(target_spec, node.params),
        )


def _validate_edges(
    report: WorkflowReport,
    snapshot: WorkflowSnapshot,
    port_types: Mapping[Tuple[str, str], str],
) -> None:
    occupied: Dict[Tuple[str, str], WorkflowEdge] = {}
    for edge in snapshot.edges:
        source = report.nodes.get(edge.source.node)
        target = report.nodes.get(edge.target.node)
        if source is None or target is None:
            continue
        source_port = _port_of(source.kind.outputs, edge.source.port)
        target_port = _port_of(target.kind.inputs, edge.target.port)
        if source_port is None:
            report.add(
                error_issue(
                    UNKNOWN_PORT,
                    f"节点 {edge.source.node} 没有输出端口 {edge.source.port!r}; "
                    f"允许: {_port_names(source.kind.outputs)}",
                    path=f"edges/{edge.id}/from/port",
                    node_id=edge.source.node,
                    edge_id=edge.id,
                )
            )
        if target_port is None:
            report.add(
                error_issue(
                    UNKNOWN_PORT,
                    f"节点 {edge.target.node} 没有输入端口 {edge.target.port!r}; "
                    f"允许: {_port_names(target.kind.inputs)}",
                    path=f"edges/{edge.id}/to/port",
                    node_id=edge.target.node,
                    edge_id=edge.id,
                )
            )
        if source_port is None or target_port is None:
            continue
        key = (edge.target.node, edge.target.port)
        if key in occupied and not target_port.multi:
            report.add(
                error_issue(
                    PORT_CONFLICT,
                    f"输入端口 {edge.target.node}.{edge.target.port} 已被 "
                    f"{occupied[key].id} 占用, 一个输入端口只能有一个来源",
                    path=f"edges/{edge.id}/to/port",
                    node_id=edge.target.node,
                    edge_id=edge.id,
                )
            )
            continue
        occupied[key] = edge
        source_type = port_types.get(
            (edge.source.node, edge.source.port), source_port.type
        )
        if not _types_compatible(source_type, target_port.type):
            report.add(
                error_issue(
                    TYPE_MISMATCH,
                    f"边 {edge.id}: {edge.source.node}.{edge.source.port} "
                    f"({source_type}) 不能连到 "
                    f"{edge.target.node}.{edge.target.port} ({target_port.type})",
                    path=f"edges/{edge.id}/to/port",
                    node_id=edge.target.node,
                    edge_id=edge.id,
                )
            )


def _validate_bindings(
    report: WorkflowReport,
    snapshot: WorkflowSnapshot,
    port_types: Mapping[Tuple[str, str], str],
) -> None:
    def sources(node: str, port: str) -> tuple[WorkflowEdge, ...]:
        return snapshot.incoming(node, port)

    def origin(node: str, port: str) -> str | None:
        edges = sources(node, port)
        if len(edges) != 1:
            return None
        source = edges[0].source.node
        while snapshot.node(source).kind == "identity":
            incoming = sources(source, "in")
            if len(incoming) != 1:
                return None
            source = incoming[0].source.node
        return source

    def fail(node: str, port: str, message: str, code: str = TYPE_MISMATCH) -> None:
        report.add(
            error_issue(code, message, path=f"nodes/{node}/inputs/{port}", node_id=node)
        )

    for node in report.of_kind("compose"):
        inputs = sources(node.id, "left") + sources(node.id, "right")
        if len(inputs) == 2:
            left, right = (
                port_types.get((e.source.node, e.source.port)) for e in inputs
            )
            if left and right and (left != right or left not in {"transform", "loss"}):
                fail(
                    node.id,
                    "right",
                    f"Compose 需要两个同类型的 transform 或 loss, 收到 {left} / {right}",
                )

    for trainer in report.of_kind("trainer"):
        model = origin(trainer.id, "model")
        optimizer = origin(trainer.id, "optimizer")
        scheduler = origin(trainer.id, "scheduler")
        if model and optimizer and origin(optimizer, "model") not in {None, model}:
            fail(trainer.id, "optimizer", "优化器必须绑定 Trainer 使用的模型")
        if (
            scheduler
            and optimizer
            and origin(scheduler, "optimizer") not in {None, optimizer}
        ):
            fail(trainer.id, "scheduler", "调度器必须绑定 Trainer 使用的优化器")

    for loader in report.of_kind("dataloader"):
        roots = _dataset_sources(snapshot, loader.id, "dataset")
        formats = {_sample_format(report.nodes.get(item)) for item in roots} - {None}
        collate = report.nodes.get(origin(loader.id, "collate") or "")
        if (
            collate is not None
            and collate.target.target == "registry:DictCollate"
            and "tuple" in formats
        ):
            fail(loader.id, "collate", "DictCollate 需要字典样本")
        for edge in snapshot.outgoing(loader.id):
            target = snapshot.node(edge.target.node)
            if target.kind == "trainer" and edge.target.port in {
                "train_loader",
                "val_loader",
            }:
                if "dict" in formats and snapshot.step_graph is None:
                    fail(loader.id, "dataset", "字典批次需要显式训练步图")
                if snapshot.execution_profile == PROFILE_DAG_RUNTIME:
                    _validate_classification_samples(
                        report, roots, origin(target.id, "model")
                    )

    for merge in report.of_kind("merge"):
        roots = _dataset_sources(snapshot, merge.id, "left") | _dataset_sources(
            snapshot, merge.id, "right"
        )
        shapes = set()
        formats = set()
        for node_id in roots:
            dataset = report.nodes.get(node_id)
            formats.add(_sample_format(dataset))
            if (
                dataset is not None
                and dataset.target.target == "registry:SyntheticClassificationDataset"
            ):
                shape = dataset.param("input_shape")
                if isinstance(shape, list) and all(type(dim) is int for dim in shape):
                    shapes.add(tuple(shape))
        if len(shapes) > 1:
            fail(merge.id, "right", "合并分支的合成样本张量形状不一致")
        if len(formats - {None}) > 1:
            fail(merge.id, "right", "合并分支的样本结构不一致")

    for split in report.of_kind("dataset_split"):
        source = report.source_of(split.id, "dataset")
        if source is not None:
            size = source.param("num_samples", source.param("size"))
            if isinstance(size, int) and size < 2:
                fail(split.id, "dataset", "数据拆分需要至少两个样本", INVALID_PARAM)


def _sample_format(node: ResolvedNode | None) -> str | None:
    if node is None:
        return None
    if node.target.target == "registry:SyntheticClassificationDataset":
        return "dict" if node.param("sample_format") == "dict" else "tuple"
    if node.target.target in {
        "torchvision.datasets:FakeData",
        "torchvision.datasets:CIFAR10",
        "torchvision.datasets:CIFAR100",
    }:
        return "tuple"
    return None


def _validate_classification_samples(
    report: WorkflowReport, roots: set[str], model_id: str | None
) -> None:
    model = report.nodes.get(model_id or "")
    if model is None:
        return
    for node_id in sorted(roots):
        source = report.nodes.get(node_id)
        if source is None:
            continue
        target = source.target.target
        input_size = model.param("input_size")
        shape = source.param("input_shape")
        if (
            model.target.target == "registry:simple_mlp"
            and target == "registry:SyntheticClassificationDataset"
            and isinstance(shape, list)
            and shape
            and all(type(dim) is int and dim > 0 for dim in shape)
            and isinstance(input_size, int)
            and prod(shape) != input_size
        ):
            report.add(
                error_issue(
                    TYPE_MISMATCH,
                    f"{node_id} 的展平特征数 {prod(shape)} 与模型 input_size {input_size} 不一致",
                    path=f"nodes/{model.id}/params/input_size",
                    node_id=model.id,
                )
            )
        classes = {
            "torchvision.datasets:CIFAR10": 10,
            "torchvision.datasets:CIFAR100": 100,
        }.get(target, source.param("num_classes"))
        outputs = model.param("num_classes")
        if isinstance(classes, int) and isinstance(outputs, int) and classes > outputs:
            report.add(
                error_issue(
                    TYPE_MISMATCH,
                    f"{node_id} 的类别数 {classes} 超过模型输出类别数 {outputs}",
                    path=f"nodes/{model.id}/params/num_classes",
                    node_id=model.id,
                )
            )


def _dataset_sources(snapshot: WorkflowSnapshot, node_id: str, port: str) -> set[str]:
    nodes = snapshot.node_map()
    pending = [edge.source.node for edge in snapshot.incoming(node_id, port)]
    seen: set[str] = set()
    result: set[str] = set()
    upstream_ports = {
        "identity": ("in",),
        "dataset_split": ("dataset",),
        "merge": ("left", "right"),
    }
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        kind = nodes[current].kind
        if kind == "dataset":
            result.add(current)
        for item in upstream_ports.get(kind, ()):
            pending.extend(
                edge.source.node for edge in snapshot.incoming(current, item)
            )
    return result


def _validate_required_inputs(
    report: WorkflowReport, snapshot: WorkflowSnapshot
) -> None:
    connected = {(edge.target.node, edge.target.port) for edge in snapshot.edges}
    for node in snapshot.nodes:
        resolved = report.nodes.get(node.id)
        if resolved is None:
            continue
        for port in resolved.kind.inputs:
            if port.required and (node.id, port.name) not in connected:
                report.add(
                    error_issue(
                        MISSING_PORT,
                        f"节点 {node.id} ({resolved.kind.label}) 的必需输入端口 "
                        f"{port.name!r} 没有连接",
                        path=f"nodes/{node.id}/inputs/{port.name}",
                        node_id=node.id,
                    )
                )


def _validate_roles(report: WorkflowReport, snapshot: WorkflowSnapshot) -> None:
    for kind in REQUIRED_ROLES:
        count = len(report.of_kind(kind))
        if count == 0:
            report.add(
                error_issue(
                    MISSING_REQUIRED_NODE,
                    f"图缺少必需的 {kind} 节点",
                    path="nodes",
                )
            )
    for kind in UNIQUE_ROLES:
        nodes = report.of_kind(kind)
        if len(nodes) > 1:
            report.add(
                error_issue(
                    DUPLICATE_ROLE,
                    f"图中有 {len(nodes)} 个 {kind} 节点, 只允许 1 个: "
                    + ", ".join(item.id for item in nodes),
                    path="nodes",
                )
            )
    for kind in OPTIONAL_UNIQUE_ROLES:
        nodes = report.of_kind(kind)
        if len(nodes) > 1:
            report.add(
                error_issue(
                    DUPLICATE_ROLE,
                    f"图中有 {len(nodes)} 个 {kind} 节点, 最多允许 1 个: "
                    + ", ".join(item.id for item in nodes),
                    path="nodes",
                )
            )
    trainers = report.of_kind("trainer")
    if len(trainers) == 1 and not snapshot.incoming(trainers[0].id, "val_loader"):
        report.add(
            warning_issue(
                MISSING_PORT,
                f"训练循环 {trainers[0].id} 没有 val_loader 输入, 训练过程中不会评估",
                path=f"nodes/{trainers[0].id}/inputs/val_loader",
                node_id=trainers[0].id,
            )
        )


def _validate_connectivity(report: WorkflowReport, snapshot: WorkflowSnapshot) -> None:
    trainers = report.of_kind("trainer")
    if len(trainers) != 1:
        return
    reachable = _connected_component(snapshot, trainers[0].id)
    for node in snapshot.nodes:
        if node.id in reachable:
            continue
        report.add(
            error_issue(
                UNREACHABLE_NODE,
                f"节点 {node.id} ({node.kind}) 与训练循环不连通, 不会影响训练",
                path=f"nodes/{node.id}",
                node_id=node.id,
            )
        )


def _validate_outputs_consumed(
    report: WorkflowReport, snapshot: WorkflowSnapshot
) -> None:
    consumed = {(edge.source.node, edge.source.port) for edge in snapshot.edges}
    trainers = {item.id for item in report.of_kind("trainer")}
    for node in snapshot.nodes:
        resolved = report.nodes.get(node.id)
        if resolved is None or not resolved.kind.outputs:
            continue
        for port in resolved.kind.outputs:
            if (node.id, port.name) in consumed:
                continue
            if node.id in trainers and port.name == "run":
                continue
            report.add(
                error_issue(
                    DANGLING_OUTPUT,
                    f"节点 {node.id} 的输出端口 {port.name!r} 没有消费方; "
                    "dag-runtime 要求显式消费或接 sink 节点",
                    path=f"nodes/{node.id}/outputs/{port.name}",
                    node_id=node.id,
                )
            )


def _connected_component(snapshot: WorkflowSnapshot, root: str) -> set[str]:
    adjacency: Dict[str, set[str]] = {node.id: set() for node in snapshot.nodes}
    for edge in snapshot.edges:
        adjacency.setdefault(edge.source.node, set()).add(edge.target.node)
        adjacency.setdefault(edge.target.node, set()).add(edge.source.node)
    seen: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency.get(current, ()))
    return seen


def _port_of(ports: Sequence[object], name: str) -> Optional[object]:
    for port in ports:
        if getattr(port, "name", None) == name:
            return port
    return None


def _port_names(ports: Sequence[object]) -> str:
    names = [getattr(port, "name", "") for port in ports]
    return ", ".join(names) or "(无)"


def _types_compatible(source_type: str, target_type: str) -> bool:
    if source_type == "any" or target_type == "any":
        return True
    return source_type == target_type


def describe_report(report: WorkflowReport) -> Dict[str, object]:
    """返回可 JSON 序列化的校验摘要, 便于日志与测试."""
    from xdl.workflow.errors import issues_to_dicts

    return {
        "ok": report.ok,
        "execution_profile": report.execution_profile,
        "issues": issues_to_dicts(report.issues),
        "warnings": issues_to_dicts(report.warnings),
        "nodes": {
            node_id: {"kind": item.kind.kind, "target": item.target.target}
            for node_id, item in sorted(report.nodes.items())
        },
    }


__all__ = [
    "PROFILE_DAG_RUNTIME",
    "PROFILE_SCHEMA_V1",
    "REQUIRED_ROLES",
    "ResolvedNode",
    "UNIQUE_ROLES",
    "WorkflowReport",
    "describe_report",
    "validate_workflow",
]
