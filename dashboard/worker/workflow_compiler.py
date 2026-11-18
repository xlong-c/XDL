"""工作流编译器: 把有类型 DAG 编译为 XDL 可执行配置.

``schema-v1`` profile 编译成现有 schema v1 映射 (再由 ``setup_from_yaml()`` 构建);
``dag-runtime`` profile 只产出运行清单, 由 :mod:`xdl.workflow.executor` 构建
``TrainSetup``.

本模块是 XDL 语义的唯一事实源: 端口归属, lowering 规则和 profile 边界都在这里,
Go dashboard 不复制这些规则.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from xdl.config import load_config_with_schema
from xdl.workflow import Issue, WorkflowError
from xdl.workflow.catalog import NodeCatalog
from xdl.workflow.errors import UNSUPPORTED_GRAPH, error_issue, issues_to_dicts

from .workflow_catalog import (
    TargetResolutionError,
    load_workflow_catalog,
    resolve_data_path,
)
from .workflow_model import CompileResult, config_bytes_hash
from .workflow_validator import (
    PROFILE_DAG_RUNTIME,
    PROFILE_SCHEMA_V1,
    ResolvedNode,
    WorkflowReport,
    validate_workflow,
)

TRANSFORM_COMPOSE_TARGET = "torchvision.transforms:Compose"
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


class WorkflowCompileError(WorkflowError):
    """编译阶段的不可恢复错误."""

    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues: List[Issue] = list(issues)
        super().__init__("; ".join(f"{i.code}: {i.message}" for i in self.issues))


@dataclass(frozen=True)
class CompileOptions:
    """编译选项."""

    data_root: Optional[str] = None
    project_root: Optional[str] = None
    output_path: Optional[str] = None
    system_overrides: Mapping[str, Any] = field(default_factory=dict)
    check_registry: bool = True
    catalog_path: Optional[str] = None


def compile_workflow(
    payload: object,
    options: Optional[CompileOptions] = None,
) -> CompileResult:
    """校验并编译一份快照.

    ``options.output_path`` 为空时只做校验与 lowering, 不写文件 (供预检使用).
    """
    resolved_options = options or CompileOptions()
    catalog = load_workflow_catalog(resolved_options.catalog_path)
    report = validate_workflow(
        payload,
        catalog=catalog,
        data_root=resolved_options.data_root,
        check_registry=resolved_options.check_registry,
    )
    if not report.ok:
        return CompileResult(
            ok=False,
            action="compile",
            execution_profile=report.execution_profile,
            issues=tuple(report.issues),
            warnings=tuple(report.warnings),
        )
    if report.snapshot is None:
        return CompileResult(
            ok=False,
            action="compile",
            execution_profile=report.execution_profile,
            issues=(error_issue(UNSUPPORTED_GRAPH, "快照未能解析为 WorkflowSnapshot"),),
        )
    try:
        if report.execution_profile == PROFILE_SCHEMA_V1:
            config, lowering_issues = _lower_schema_v1(
                report, catalog, resolved_options
            )
        elif report.execution_profile == PROFILE_DAG_RUNTIME:
            config, lowering_issues = _lower_dag_runtime(report)
        else:
            config, lowering_issues = (
                None,
                [
                    error_issue(
                        UNSUPPORTED_GRAPH,
                        f"不支持的执行 profile {report.execution_profile!r}",
                    )
                ],
            )
    except WorkflowCompileError as exc:
        return CompileResult(
            ok=False,
            action="compile",
            execution_profile=report.execution_profile,
            issues=tuple(exc.issues),
            warnings=tuple(report.warnings),
        )
    if lowering_issues:
        return CompileResult(
            ok=False,
            action="compile",
            execution_profile=report.execution_profile,
            issues=tuple(lowering_issues),
            warnings=tuple(report.warnings),
        )

    config_path = ""
    config_hash = ""
    if config is not None and resolved_options.output_path:
        written = _write_config(config, resolved_options.output_path)
        config_path = str(written)
        config_hash = config_bytes_hash(written.read_bytes())
    return CompileResult(
        ok=True,
        action="compile",
        execution_profile=report.execution_profile,
        issues=tuple(report.issues),
        warnings=tuple(report.warnings),
        config=config,
        config_path=config_path,
        config_hash=config_hash,
        runtime_manifest=_runtime_manifest(report)
        if report.execution_profile == PROFILE_DAG_RUNTIME
        else {},
    )


def validate_only(
    payload: object, options: Optional[CompileOptions] = None
) -> CompileResult:
    """只做校验与 lowering 演练, 不写任何文件."""
    resolved = options or CompileOptions()
    return compile_workflow(
        payload,
        CompileOptions(
            data_root=resolved.data_root,
            project_root=resolved.project_root,
            output_path=None,
            system_overrides=resolved.system_overrides,
            check_registry=resolved.check_registry,
            catalog_path=resolved.catalog_path,
        ),
    )


def _lower_schema_v1(
    report: WorkflowReport,
    catalog: NodeCatalog,
    options: CompileOptions,
) -> Tuple[Optional[Dict[str, Any]], List[Issue]]:
    issues: List[Issue] = []
    snapshot = report.snapshot
    assert snapshot is not None
    trainer = report.of_kind("trainer")[0]

    config: Dict[str, Any] = {"config_version": 1}
    trainer_config = _trainer_config(trainer)
    if trainer_config:
        config["trainer"] = trainer_config

    seed = _runtime_seed(snapshot.settings)
    if seed is not None:
        config.setdefault("runtime", {})["seed"] = seed

    for role, port in (("train", "train_loader"), ("val", "val_loader")):
        loader = report.source_of(trainer.id, port)
        if loader is None:
            continue
        section_issues = _lower_loader_branch(report, role, loader, config, options)
        issues.extend(section_issues)

    model = report.source_of(trainer.id, "model")
    if model is None:
        issues.append(
            error_issue(
                UNSUPPORTED_GRAPH,
                f"训练循环 {trainer.id} 的 model 输入无法解析",
                node_id=trainer.id,
            )
        )
    else:
        config["model"] = _component_config(model, options, issues)

    losses = _sorted_nodes(report.input_nodes(trainer.id, "loss"))
    if not losses:
        issues.append(
            error_issue(
                UNSUPPORTED_GRAPH,
                f"训练循环 {trainer.id} 的 loss 输入无法解析",
                node_id=trainer.id,
            )
        )
    else:
        config["loss"] = [_component_config(node, options, issues) for node in losses]

    optimizer = report.source_of(trainer.id, "optimizer")
    optimization: Dict[str, Any] = {}
    if optimizer is None:
        issues.append(
            error_issue(
                UNSUPPORTED_GRAPH,
                f"训练循环 {trainer.id} 的 optimizer 输入无法解析",
                node_id=trainer.id,
            )
        )
    else:
        optimization["optimizer"] = _component_config(optimizer, options, issues)
        scheduler = report.source_of(trainer.id, "scheduler")
        if scheduler is not None:
            optimization["scheduler"] = _component_config(scheduler, options, issues)
    if optimization:
        config["optimization"] = optimization

    metrics = [
        _component_config(node, options, issues) for node in report.of_kind("metric")
    ]
    if metrics:
        config["metrics"] = metrics

    checkpoints = report.of_kind("checkpoint")
    if checkpoints:
        checkpoint_params = {
            name: checkpoints[0].param(name)
            for name in CHECKPOINT_PARAM_NAMES
            if checkpoints[0].param(name) is not None
        }
        if checkpoint_params:
            config["checkpoint"] = checkpoint_params

    if issues:
        return None, issues
    merged, merge_issues = _apply_system_overrides(config, options.system_overrides)
    issues.extend(merge_issues)
    if issues:
        return None, issues
    validation_issues = _validate_schema_v1(merged)
    issues.extend(validation_issues)
    if issues:
        return None, issues
    return _to_plain_mapping(merged), issues


def _lower_loader_branch(
    report: WorkflowReport,
    role: str,
    loader: ResolvedNode,
    config: Dict[str, Any],
    options: CompileOptions,
) -> List[Issue]:
    issues: List[Issue] = []
    dataset = report.source_of(loader.id, "dataset")
    if dataset is None:
        return [
            error_issue(
                UNSUPPORTED_GRAPH,
                f"{role} dataloader {loader.id} 没有可解析的 dataset 输入",
                node_id=loader.id,
            )
        ]
    transform_chain = _transform_chain(report, dataset)
    dataset_config = _component_config(dataset, options, issues)
    if transform_chain:
        config[f"{role}_transforms"] = {
            "target": TRANSFORM_COMPOSE_TARGET,
            "params": {
                "transforms": [
                    _component_config(node, options, issues) for node in transform_chain
                ]
            },
        }
        params = dict(dataset_config.get("params") or {})
        params["transform"] = f"${{{role}_transforms}}"
        dataset_config["params"] = params
    config[f"{role}_dataset"] = dataset_config
    config[f"{role}_dataloader"] = {
        "dataset": f"${{{role}_dataset}}",
        "params": {
            name: loader.param(name)
            for name in (
                "batch_size",
                "num_workers",
                "shuffle",
                "drop_last",
                "pin_memory",
            )
            if loader.param(name) is not None
        },
    }
    return issues


def _transform_chain(
    report: WorkflowReport, dataset: ResolvedNode
) -> List[ResolvedNode]:
    chain: List[ResolvedNode] = []
    current = report.source_of(dataset.id, "transform")
    while current is not None:
        chain.append(current)
        current = report.source_of(current.id, "pipeline")
    chain.reverse()
    return chain


def _component_config(
    node: ResolvedNode, options: CompileOptions, issues: List[Issue]
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for spec in node.target.params:
        value = node.params.get(spec.name)
        if value is None:
            continue
        if spec.type == "data_path":
            try:
                value = resolve_data_path(str(value), options.data_root)
            except TargetResolutionError as exc:
                issues.append(
                    error_issue(
                        "path_not_allowed",
                        f"节点 {node.id} 的参数 {spec.name}: {exc}",
                        path=f"nodes/{node.id}/params/{spec.name}",
                        node_id=node.id,
                    )
                )
                continue
        params[spec.name] = value
    return {"target": node.target.target, "params": params}


def _trainer_config(trainer: ResolvedNode) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name in TRAINER_PARAM_NAMES:
        value = trainer.param(name)
        if value is not None:
            result[name] = value
    return result


def _runtime_seed(settings: Mapping[str, Any]) -> Optional[int]:
    runtime = settings.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    seed = runtime.get("seed")
    if seed is None:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise WorkflowCompileError(
            [
                error_issue(
                    "invalid_snapshot",
                    "settings.runtime.seed 必须是整数",
                    path="settings/runtime/seed",
                )
            ]
        )
    return seed


def _lower_dag_runtime(
    report: WorkflowReport,
) -> Tuple[Optional[Dict[str, Any]], List[Issue]]:
    from xdl.workflow.executor import SUPPORTED_KINDS

    unsupported = sorted(
        {
            node.kind.kind
            for node in report.nodes.values()
            if node.kind.kind not in SUPPORTED_KINDS
        }
    )
    if unsupported:
        return None, [
            error_issue(
                UNSUPPORTED_GRAPH,
                "dag-runtime 尚不支持节点类型: " + ", ".join(unsupported),
            )
        ]
    return None, []


def _apply_system_overrides(
    config: Dict[str, Any], overrides: Mapping[str, Any]
) -> Tuple[DictConfig, List[Issue]]:
    merged = OmegaConf.create(config)
    for path, value in overrides.items():
        OmegaConf.update(merged, str(path), value, merge=False)
    return merged, []


def _validate_schema_v1(config: DictConfig) -> List[Issue]:
    try:
        load_config_with_schema(config, resolve=True)
    except Exception as exc:  # noqa: BLE001 - 统一映射为可定位问题
        return [
            error_issue(
                UNSUPPORTED_GRAPH,
                f"生成的 schema v1 配置未通过 XDL 校验: {exc}",
                path="config",
            )
        ]
    return []


def _to_plain_mapping(config: DictConfig) -> Dict[str, Any]:
    plain = OmegaConf.to_container(config, resolve=False)
    if not isinstance(plain, dict):
        raise WorkflowCompileError(
            [error_issue(UNSUPPORTED_GRAPH, "编译结果必须是映射")]
        )
    return plain


def _write_config(config: Mapping[str, Any], output_path: str) -> Path:
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(dict(config)), f=str(destination))
    load_config_with_schema(destination, resolve=True)
    return destination


def _runtime_manifest(report: WorkflowReport) -> Dict[str, Any]:
    snapshot = report.snapshot
    assert snapshot is not None
    return {
        "profile": report.execution_profile,
        "topological_order": list(snapshot.topological_order()),
        "nodes": {
            node_id: {"kind": item.kind.kind, "target": item.target.target}
            for node_id, item in sorted(report.nodes.items())
        },
    }


def _sorted_nodes(nodes: Sequence[ResolvedNode]) -> List[ResolvedNode]:
    return sorted(nodes, key=lambda item: item.id)


def result_payload(result: CompileResult) -> Dict[str, Any]:
    """返回 CLI 响应用的结果字典."""
    return result.as_dict()


def issues_payload(issues: Sequence[Issue]) -> List[Dict[str, Any]]:
    """返回问题的字典列表."""
    return issues_to_dicts(issues)


__all__ = [
    "CHECKPOINT_PARAM_NAMES",
    "TRAINER_PARAM_NAMES",
    "TRANSFORM_COMPOSE_TARGET",
    "CompileOptions",
    "WorkflowCompileError",
    "compile_workflow",
    "issues_payload",
    "result_payload",
    "validate_only",
]
