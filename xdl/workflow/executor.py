"""dag-runtime 执行器: 按拓扑序把有类型 DAG 构建成 ``TrainSetup``.

设计边界:

- 只构建组件, 不实现训练循环. 训练生命周期继续由 ``Trainer`` 负责, 执行器只产出
  ``TrainSetup``, 因此 checkpoint, pause/resume 和 DashboardCallback 与 schema-v1
  profile 完全一致.
- 节点构建复用 ``xdl.config.builder``, 不重复实现 target 解析.
- 任何不支持的节点类型在构建前被拒绝, 不进入半执行状态.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, IterableDataset, default_collate
from torchvision.transforms import Compose

from xdl.config import (
    CheckpointConfig,
    LoggingConfig,
    RuntimeConfig,
    TrainSetup,
    TrainerConfig,
    merge_with_schema,
    resolve_config,
    to_plain_dict,
)
from xdl.config.schema import DiagnosticsConfig
from xdl.config.builder import (
    build_dataloader,
    build_collate_fn,
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_transform,
)
from xdl.config.loss_weighted import WeightedLoss
from xdl.dataset.split import train_val_split

from .catalog import NodeCatalog, assembly_port_types, load_catalog
from .errors import (
    UNSUPPORTED_GRAPH,
    WorkflowExecutionError,
    WorkflowValidationError,
    error_issue,
)
from .ir import WorkflowNode, WorkflowSnapshot
from .step_graph import StepGraphModel, compile_step_graph

SUPPORTED_KINDS = frozenset(
    {
        "transform",
        "dataset",
        "dataset_split",
        "collate",
        "dataloader",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "metric",
        "trainer",
        "checkpoint",
        "identity",
        "compose",
        "merge",
        "sink",
    }
)

TRAINER_PARAM_NAMES: Tuple[str, ...] = (
    "max_epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "grad_clip_max_norm",
    "precision",
)
CHECKPOINT_PARAM_NAMES: Tuple[str, ...] = (
    "save_last",
    "save_top_k",
    "monitor",
    "mode",
    "every_n_epochs",
)
LOADER_PARAM_NAMES: Tuple[str, ...] = (
    "batch_size",
    "num_workers",
    "shuffle",
    "drop_last",
    "pin_memory",
)


@dataclass(frozen=True)
class WorkflowRuntimeResult:
    """一次 dag-runtime 构建的结果."""

    setup: TrainSetup
    order: Tuple[str, ...]
    execution_profile: str = "dag-runtime"


class WorkflowExecutor:
    """把有类型 DAG 构建为 ``TrainSetup``."""

    def __init__(self, catalog: Optional[NodeCatalog] = None) -> None:
        self.catalog = catalog or load_catalog()

    def build_setup(
        self,
        snapshot: WorkflowSnapshot,
        *,
        system_overrides: Optional[Mapping[str, Any]] = None,
    ) -> TrainSetup:
        """按拓扑序构建并返回 ``TrainSetup``."""
        return self.execute(snapshot, system_overrides=system_overrides).setup

    def execute(
        self,
        snapshot: WorkflowSnapshot,
        *,
        system_overrides: Optional[Mapping[str, Any]] = None,
    ) -> WorkflowRuntimeResult:
        """执行图构建, 返回带拓扑序的结果对象."""
        self._check_supported(snapshot)
        program = compile_step_graph(snapshot, self.catalog)
        order = snapshot.topological_order()
        port_types = assembly_port_types(snapshot, self.catalog)
        outputs: Dict[str, Dict[str, Any]] = {}
        setup: Optional[TrainSetup] = None
        for node_id in order:
            node = snapshot.node(node_id)
            inputs = _gather_inputs(snapshot, outputs, node_id)
            if node.kind == "trainer":
                setup = self._build_trainer(
                    node, inputs, snapshot, system_overrides or {}
                )
                outputs[node_id] = {"run": setup}
            elif node.kind == "sink":
                outputs[node_id] = {}
            else:
                outputs[node_id] = self._evaluate(snapshot, node, inputs, port_types)
        if setup is None:
            raise WorkflowExecutionError("图缺少 trainer 节点, 无法构建 TrainSetup")
        if program is not None:
            if setup.optimizer is None:
                raise WorkflowExecutionError(
                    "训练步图缺少 optimizer, 无法构建 StepGraphModel"
                )
            losses = {
                item.id: outputs[item.id]["loss"]
                for item in snapshot.nodes_of_kind("loss")
            }
            metrics = {
                item.id: outputs[item.id]["metrics"][0]
                for item in snapshot.nodes_of_kind("metric")
            }
            setup.model = StepGraphModel(
                program, setup.model, setup.optimizer, setup.scheduler, losses, metrics
            )
        return WorkflowRuntimeResult(setup=setup, order=order)

    def _check_supported(self, snapshot: WorkflowSnapshot) -> None:
        unsupported = sorted(
            {node.kind for node in snapshot.nodes if node.kind not in SUPPORTED_KINDS}
        )
        if unsupported:
            raise WorkflowValidationError(
                [
                    error_issue(
                        UNSUPPORTED_GRAPH,
                        "dag-runtime 不支持节点类型: " + ", ".join(unsupported),
                    )
                ]
            )

    def _evaluate(
        self,
        snapshot: WorkflowSnapshot,
        node: WorkflowNode,
        inputs: Dict[str, List[Any]],
        port_types: Mapping[Tuple[str, str], str],
    ) -> Dict[str, Any]:
        params = dict(node.params)
        component = {"target": node.target, "params": params}
        if node.kind == "transform":
            operation = build_transform(component)
            previous = _single(inputs, "pipeline")
            return {
                "pipeline": operation
                if previous is None
                else Compose([previous, operation])
            }
        if node.kind == "dataset":
            return {
                "samples": build_dataset(
                    component, transform=_single(inputs, "transform")
                )
            }
        if node.kind == "dataset_split":
            dataset = _require(_single(inputs, "dataset"), node, "dataset")
            if (
                isinstance(dataset, IterableDataset)
                or not hasattr(dataset, "__getitem__")
                or len(dataset) < 2
            ):
                raise WorkflowExecutionError(
                    f"节点 {node.id} 需要至少两个可按索引读取的样本"
                )
            seed = params.get("seed", _runtime_seed(snapshot))
            train, val = train_val_split(
                dataset, params.get("val_ratio", 0.1), seed=42 if seed is None else seed
            )
            return {"train": train, "val": val}
        if node.kind == "collate":
            fn = (
                default_collate
                if node.target == "builtin:DefaultCollate"
                else build_collate_fn(component)
            )
            return {"collate": fn}
        if node.kind == "dataloader":
            dataset = _require(_single(inputs, "dataset"), node, "dataset")
            loader_params = {
                name: params[name] for name in LOADER_PARAM_NAMES if name in params
            }
            # 显式 loader 参数覆盖旧图的 trainer.batch_size 默认来源。
            batch_size = _trainer_batch_size(snapshot)
            if batch_size is not None:
                loader_params.setdefault("batch_size", batch_size)
            return {
                "batches": build_dataloader(
                    dataset,
                    {"params": loader_params},
                    collate_fn=_single(inputs, "collate"),
                )
            }
        if node.kind == "model":
            return {"module": build_model(component)}
        if node.kind == "loss":
            return {"loss": build_loss([component])}
        if node.kind == "optimizer":
            model = _require(_single(inputs, "model"), node, "model")
            return {"optimizer": build_optimizer(model, component)}
        if node.kind == "scheduler":
            optimizer = _require(_single(inputs, "optimizer"), node, "optimizer")
            return {"scheduler": build_scheduler(optimizer, component)}
        if node.kind == "metric":
            return {"metrics": build_metrics([component])}
        if node.kind == "checkpoint":
            return {"checkpoint": params}
        if node.kind == "identity":
            return {"out": _require(_single(inputs, "in"), node, "in")}
        if node.kind == "compose":
            left = _require(_single(inputs, "left"), node, "left")
            right = _require(_single(inputs, "right"), node, "right")
            return {"out": _compose_values(snapshot, node, left, right, port_types)}
        if node.kind == "merge":
            left = _require(_single(inputs, "left"), node, "left")
            right = _require(_single(inputs, "right"), node, "right")
            return {"out": _merge_datasets(node, left, right)}
        raise WorkflowExecutionError(f"节点 {node.id} 的类型 {node.kind} 未实现")

    def _build_trainer(
        self,
        node: WorkflowNode,
        inputs: Dict[str, List[Any]],
        snapshot: WorkflowSnapshot,
        system_overrides: Mapping[str, Any],
    ) -> TrainSetup:
        params = dict(node.params)
        trainer_params = {
            name: params[name] for name in TRAINER_PARAM_NAMES if name in params
        }
        train_loader = _require(_single(inputs, "train_loader"), node, "train_loader")
        model = _require(_single(inputs, "model"), node, "model")
        optimizer = _require(_single(inputs, "optimizer"), node, "optimizer")
        losses = [item for item in inputs.get("loss", []) if item is not None]
        if not losses:
            raise WorkflowExecutionError(f"节点 {node.id} 缺少 loss 输入")
        loss_fn = losses[0] if len(losses) == 1 else WeightedLoss(losses)
        checkpoint_params = _checkpoint_params(snapshot)
        seed = _runtime_seed(snapshot)
        merged = _merge_structured_config(
            trainer_params=trainer_params,
            seed=seed,
            checkpoint_params=checkpoint_params,
            system_overrides=system_overrides,
        )
        return TrainSetup(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            val_loader=_single(inputs, "val_loader"),
            scheduler=_single(inputs, "scheduler"),
            metrics=_build_metrics(snapshot),
            callbacks=[],
            trainer=_to_object(merged.trainer, TrainerConfig),
            runtime=_to_object(merged.runtime, RuntimeConfig),
            logging=_to_object(merged.logging, LoggingConfig),
            checkpoint=_to_object(merged.checkpoint, CheckpointConfig),
            diagnostics=_to_object(merged.diagnostics, DiagnosticsConfig),
            accelerate=None,
            deepspeed=None,
            full_config=to_plain_dict(resolve_config(merged)),
        )


def _merge_structured_config(
    *,
    trainer_params: Mapping[str, Any],
    seed: Optional[int],
    checkpoint_params: Mapping[str, Any],
    system_overrides: Mapping[str, Any],
) -> Any:
    """合并 schema v1 结构化配置与 dashboard 系统覆盖."""
    base: Dict[str, Any] = {"config_version": 1, "trainer": dict(trainer_params)}
    if seed is not None:
        base["runtime"] = {"seed": seed}
    if checkpoint_params:
        base["checkpoint"] = dict(checkpoint_params)
    merged = merge_with_schema(base)
    for path, value in system_overrides.items():
        OmegaConf.update(merged, str(path), value, merge=False)
    return merged


def _to_object(node: Any, expected: type) -> Any:
    value = OmegaConf.to_object(node)
    if not isinstance(value, expected):
        raise WorkflowExecutionError(
            f"expected {expected.__name__}, got {type(value).__name__}"
        )
    return value


def _gather_inputs(
    snapshot: WorkflowSnapshot,
    outputs: Mapping[str, Dict[str, Any]],
    node_id: str,
) -> Dict[str, List[Any]]:
    gathered: Dict[str, List[Any]] = {}
    for edge in snapshot.incoming(node_id):
        source_output = outputs.get(edge.source.node, {})
        if edge.source.port not in source_output:
            raise WorkflowExecutionError(
                f"边 {edge.id} 的源端口 {edge.source.node}.{edge.source.port} 没有产出"
            )
        gathered.setdefault(edge.target.port, []).append(
            source_output[edge.source.port]
        )
    return gathered


def _single(inputs: Mapping[str, List[Any]], port: str) -> Any:
    values = inputs.get(port)
    if not values:
        return None
    return values[0]


def _require(value: Any, node: WorkflowNode, port: str) -> Any:
    if value is None:
        raise WorkflowExecutionError(f"节点 {node.id} 的输入端口 {port} 没有值")
    return value


def _compose_values(
    snapshot: WorkflowSnapshot,
    node: WorkflowNode,
    left: Any,
    right: Any,
    port_types: Mapping[Tuple[str, str], str],
) -> Any:
    """按上游节点类型组合两个值.

    不能靠 ``isinstance(x, nn.Module)`` 判断: torchvision 的 transform 同样是
    ``nn.Module``. 组合语义必须来自图的声明类型, 而不是运行时猜类型.
    """
    left_kind = _upstream_type(snapshot, node, "left", port_types)
    right_kind = _upstream_type(snapshot, node, "right", port_types)
    if left_kind != right_kind:
        raise WorkflowExecutionError(
            f"节点 {node.id} 的两个输入类型不同: {left_kind!r} 与 {right_kind!r}"
        )
    if left_kind == "transform":
        return Compose([left, right])
    if left_kind == "loss":
        return WeightedLoss([left, right], [1.0, 1.0])
    raise WorkflowExecutionError(f"节点 {node.id} 不支持组合 {left_kind!r} 类型的值")


def _upstream_type(
    snapshot: WorkflowSnapshot,
    node: WorkflowNode,
    port: str,
    port_types: Mapping[Tuple[str, str], str],
) -> str:
    edges = snapshot.incoming(node.id, port)
    if not edges:
        raise WorkflowExecutionError(f"节点 {node.id} 的输入端口 {port} 没有来源")
    source = edges[0].source
    return port_types.get((source.node, source.port), "any")


def _merge_datasets(node: WorkflowNode, left: Any, right: Any) -> Any:
    if any(
        isinstance(value, IterableDataset)
        or not hasattr(value, "__len__")
        or not hasattr(value, "__getitem__")
        for value in (left, right)
    ):
        raise WorkflowExecutionError(
            f"节点 {node.id} 只能合并数据集, 收到 {type(left).__name__} 与 "
            f"{type(right).__name__}"
        )
    strategy = dict(node.params).get("strategy", "concat")
    if strategy != "concat":
        raise WorkflowExecutionError(f"节点 {node.id} 不支持合并策略 {strategy!r}")
    return ConcatDataset([left, right])


def _build_metrics(snapshot: WorkflowSnapshot) -> List[Any]:
    """按节点 ID 顺序构建全部 metric 节点.

    metric 节点通过 ``run`` 输入表达"属于这次运行", 但指标对象必须在 trainer 构建
    时就绪, 因此这里直接按快照声明构建, 不依赖拓扑求值顺序.
    """
    configs: List[Dict[str, Any]] = []
    for node in sorted(snapshot.nodes, key=lambda item: item.id):
        if node.kind != "metric":
            continue
        configs.append({"target": node.target, "params": dict(node.params)})
    return build_metrics(configs)


def _trainer_batch_size(snapshot: WorkflowSnapshot) -> Optional[int]:
    for node in snapshot.nodes:
        if node.kind != "trainer":
            continue
        value = node.params.get("batch_size")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _checkpoint_params(snapshot: WorkflowSnapshot) -> Dict[str, Any]:
    for node in snapshot.nodes:
        if node.kind != "checkpoint":
            continue
        return {
            name: node.params[name]
            for name in CHECKPOINT_PARAM_NAMES
            if name in node.params
        }
    return {}


def _runtime_seed(snapshot: WorkflowSnapshot) -> Optional[int]:
    runtime = snapshot.settings.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    seed = runtime.get("seed")
    if seed is None:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise WorkflowExecutionError("settings.runtime.seed must be an integer")
    return int(seed)


__all__ = [
    "CHECKPOINT_PARAM_NAMES",
    "LOADER_PARAM_NAMES",
    "SUPPORTED_KINDS",
    "TRAINER_PARAM_NAMES",
    "WorkflowExecutor",
    "WorkflowRuntimeResult",
]
