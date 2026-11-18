"""Compile and evaluate the calculation for one batch."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from xdl.trainer.core_model import CoreModel

from .catalog import NodeCatalog, apply_param_defaults, validate_params
from .errors import (
    DUPLICATE_ROLE,
    INVALID_PARAM,
    MISSING_PORT,
    MISSING_REQUIRED_NODE,
    PORT_CONFLICT,
    TYPE_MISMATCH,
    UNKNOWN_NODE_KIND,
    UNKNOWN_PORT,
    UNKNOWN_TARGET,
    UNREACHABLE_NODE,
    UNSUPPORTED_PROFILE,
    Issue,
    WorkflowExecutionError,
    WorkflowValidationError,
    error_issue,
)
from .ir import WorkflowNode, WorkflowSnapshot


@dataclass(frozen=True)
class StepProgram:
    """Ordered operations and their already-resolved input references."""

    operations: tuple[WorkflowNode, ...]
    inputs: Mapping[str, Mapping[str, tuple[tuple[str, str], ...]]]

    def evaluate(
        self,
        batch: Any,
        model: torch.nn.Module,
        losses: Mapping[str, torch.nn.Module],
        metrics: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for node in self.operations:
            inputs = {
                port: [values[source][output] for source, output in refs]
                for port, refs in self.inputs[node.id].items()
            }
            try:
                if node.kind == "StepOutput":
                    return _scalar(inputs["loss"][0]), dict(inputs.get("metrics", []))
                values[node.id] = _evaluate(node, inputs, batch, model, losses, metrics)
            except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                raise WorkflowExecutionError(
                    f"step_graph/nodes/{node.id} ({node.kind}): {exc}"
                ) from exc
        raise WorkflowExecutionError("step_graph: missing StepOutput")


def compile_step_graph(
    snapshot: WorkflowSnapshot, catalog: NodeCatalog
) -> StepProgram | None:
    graph = snapshot.step_graph
    if graph is None:
        return None
    if snapshot.execution_profile != "dag-runtime":
        raise WorkflowValidationError(
            [
                error_issue(
                    UNSUPPORTED_PROFILE,
                    "step_graph 只支持 dag-runtime",
                    path="step_graph",
                )
            ]
        )
    issues: list[Issue] = []
    outer = snapshot.node_map()
    trainer = snapshot.nodes_of_kind("trainer")
    loss_ids = (
        {edge.source.node for edge in snapshot.incoming(trainer[0].id, "loss")}
        if len(trainer) == 1
        else set()
    )
    specs = {}
    resolved = {}
    counts = Counter(node.kind for node in graph.nodes)
    for role in ("BatchInput", "StepOutput"):
        if counts[role] != 1:
            issues.append(
                error_issue(
                    MISSING_REQUIRED_NODE if not counts[role] else DUPLICATE_ROLE,
                    f"训练步图必须恰有一个 {role}",
                    path="step_graph/nodes",
                )
            )
    metric_names: set[str] = set()
    for node in graph.nodes:
        spec = catalog.step_kinds.get(node.kind)
        if spec is None:
            issues.append(
                _node_issue(UNKNOWN_NODE_KIND, node, "未知训练步节点", "kind")
            )
            continue
        target = next(
            (item for item in spec.targets if item.target == node.target), None
        )
        if target is None:
            issues.append(
                _node_issue(UNKNOWN_TARGET, node, "未知训练步 target", "target")
            )
            continue
        specs[node.id] = spec
        issues.extend(
            validate_params(
                target,
                node.params,
                node_id=node.id,
                path_prefix=f"step_graph/nodes/{node.id}/params",
            )
        )
        params = apply_param_defaults(target, node.params)
        resolved[node.id] = WorkflowNode(node.id, node.kind, node.target, params)
        if node.kind == "Select" and (
            ("index" in params) == ("key" in params) or params.get("key") == ""
        ):
            issues.append(
                _node_issue(
                    INVALID_PARAM, node, "必须指定 index 或非空 key, 二选一", "params"
                )
            )
        if node.kind in {"LossCall", "MetricCall"}:
            ref = params.get("component")
            component = outer.get(ref) if isinstance(ref, str) else None
            expected = "loss" if node.kind == "LossCall" else "metric"
            if (
                component is None
                or component.kind != expected
                or (expected == "loss" and ref not in loss_ids)
            ):
                issues.append(
                    _node_issue(
                        INVALID_PARAM,
                        node,
                        f"必须引用已装配的 {expected} 节点",
                        "params/component",
                    )
                )
        if node.kind == "MetricCall":
            name = params.get("name")
            if (
                not isinstance(name, str)
                or not name.strip()
                or name == "loss"
                or name in metric_names
            ):
                issues.append(
                    _node_issue(
                        INVALID_PARAM,
                        node,
                        "指标名称必须非空、唯一且不为 loss",
                        "params/name",
                    )
                )
            else:
                metric_names.add(name)

    inputs: dict[str, dict[str, list[tuple[str, str]]]] = {
        node.id: {} for node in graph.nodes
    }
    for edge in graph.edges:
        source = specs.get(edge.source.node)
        target_spec = specs.get(edge.target.node)
        if source is None or target_spec is None:
            continue
        output = next((p for p in source.outputs if p.name == edge.source.port), None)
        port = next((p for p in target_spec.inputs if p.name == edge.target.port), None)
        path = f"step_graph/edges/{edge.id}"
        if output is None or port is None:
            issues.append(
                error_issue(UNKNOWN_PORT, "未知训练步端口", path=path, edge_id=edge.id)
            )
            continue
        if output.type != port.type:
            issues.append(
                error_issue(
                    TYPE_MISMATCH,
                    f"{output.type} 不能连接 {port.type}",
                    path=path,
                    edge_id=edge.id,
                )
            )
            continue
        refs = inputs[edge.target.node].setdefault(port.name, [])
        if refs and not port.multi:
            issues.append(
                error_issue(PORT_CONFLICT, "输入已有来源", path=path, edge_id=edge.id)
            )
        refs.append((edge.source.node, edge.source.port))
    for node in graph.nodes:
        spec = specs.get(node.id)
        if spec is not None:
            for port in spec.inputs:
                if port.required and port.name not in inputs[node.id]:
                    issues.append(
                        _node_issue(
                            MISSING_PORT,
                            node,
                            f"缺少输入 {port.name}",
                            f"inputs/{port.name}",
                        )
                    )
    view = WorkflowSnapshot(
        snapshot.version, "", "", snapshot.execution_profile, graph.nodes, graph.edges
    )
    sinks = view.nodes_of_kind("StepOutput")
    if len(sinks) == 1:
        reachable = {*view.upstream_ids(sinks[0].id), sinks[0].id}
        for node in graph.nodes:
            if node.id not in reachable:
                issues.append(_node_issue(UNREACHABLE_NODE, node, "节点未参与步输出"))
    if issues:
        raise WorkflowValidationError(issues)
    return StepProgram(
        tuple(resolved[node_id] for node_id in view.topological_order()),
        {
            node: {port: tuple(refs) for port, refs in ports.items()}
            for node, ports in inputs.items()
        },
    )


def _node_issue(code: str, node: WorkflowNode, message: str, suffix: str = "") -> Issue:
    path = f"step_graph/nodes/{node.id}"
    return error_issue(
        code, message, node_id=node.id, path=f"{path}/{suffix}" if suffix else path
    )


def _scalar(value: Any) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 0
        or not value.is_floating_point()
    ):
        raise ValueError("loss 必须是浮点标量张量")
    return value


def _evaluate(
    node: WorkflowNode,
    inputs: Mapping[str, list[Any]],
    batch: Any,
    model: torch.nn.Module,
    losses: Mapping[str, torch.nn.Module],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    params = node.params
    if node.kind == "BatchInput":
        return {"batch": batch}
    if node.kind == "Select":
        container = inputs["batch"][0]
        if "key" in params and isinstance(container, Mapping):
            value = container[params["key"]]
        elif "index" in params and isinstance(container, (tuple, list)):
            value = container[params["index"]]
        else:
            raise TypeError("key 需要字典 batch, index 需要 tuple/list batch")
        if not isinstance(value, torch.Tensor):
            raise TypeError("所选字段必须是张量")
        return {"value": value}
    if node.kind == "Forward":
        prediction = model(inputs["features"][0])
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("模型必须返回张量")
        return {"predictions": prediction}
    if node.kind == "LossCall":
        return {
            "loss": _scalar(
                losses[params["component"]](
                    inputs["predictions"][0], inputs["labels"][0]
                )
            )
        }
    if node.kind == "MetricCall":
        with torch.no_grad():
            value = metrics[params["component"]](
                inputs["predictions"][0].detach(), inputs["labels"][0].detach()
            )
        return {"value": (params["name"], value)}
    if node.kind == "WeightedSum":
        return {
            "loss": params["left_weight"] * inputs["left"][0]
            + params["right_weight"] * inputs["right"][0]
        }
    raise WorkflowExecutionError(f"未知训练步节点: {node.kind}")


class StepGraphModel(CoreModel):
    """Delegate batch calculations to a graph, retaining CoreModel optimization."""

    def __init__(
        self,
        program: StepProgram,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        losses: Mapping[str, torch.nn.Module],
        metrics: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.program = program
        self.model = model
        self.loss_modules = torch.nn.ModuleList(losses.values())
        self.loss_refs = tuple(losses)
        self.metric_functions = dict(metrics)
        self.metric_modules = torch.nn.ModuleList(
            value for value in metrics.values() if isinstance(value, torch.nn.Module)
        )
        self._external_optimizer = optimizer
        self._external_scheduler = scheduler

    def configure_optimizers(self) -> Any:
        if self._external_scheduler is not None:
            return [self._external_optimizer], [self._external_scheduler]
        return self._external_optimizer

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def _compute(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.program.evaluate(
            batch,
            self.model,
            dict(zip(self.loss_refs, self.loss_modules, strict=True)),
            self.metric_functions,
        )

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        loss, metrics = self._compute(batch)
        if self.manual_optimization_step(loss):
            for scheduler in self._schedules:
                scheduler.step()
        self.log("loss", loss.detach(), prefix="train")
        for name, value in metrics.items():
            self.log(name, value, prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        loss, metrics = self._compute(batch)
        self.log("loss", loss.detach(), prefix="val")
        for name, value in metrics.items():
            self.log(name, value, prefix="val")
