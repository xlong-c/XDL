"""Workflow compiler and schema-v1 lowering tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import pytest

from xdl.config import setup_from_yaml

from dashboard.worker.workflow_compiler import (
    CompileOptions,
    compile_workflow,
    validate_only,
)

from .conftest import REPO_ROOT

DEMO_CONFIG = REPO_ROOT / "config/vgg_cifar10_dashboard_demo.yaml"
SYSTEM_OVERRIDES = {
    "runtime.output_dir": "/tmp/xdl-workflow-test",
    "runtime.device": "cpu",
    "logging.log_dir": "/tmp/xdl-workflow-test/logs",
    "logging.enable_console": False,
    "logging.enable_tensorboard": False,
    "checkpoint.dirpath": "/tmp/xdl-workflow-test/checkpoints",
}


def _options(tmp_path: Path | None = None, **kwargs: Any) -> CompileOptions:
    return CompileOptions(
        data_root=str(REPO_ROOT / "data"),
        project_root=str(REPO_ROOT),
        output_path=None if tmp_path is None else str(tmp_path / "config.yaml"),
        system_overrides=dict(SYSTEM_OVERRIDES),
        **kwargs,
    )


def _node(payload: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    for node in payload["nodes"]:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"node {node_id} not found")


def _resolved_config(path: Path) -> Dict[str, Any]:
    from xdl.config import load_config_with_schema, to_plain_dict

    return to_plain_dict(load_config_with_schema(path, resolve=True), resolve=True)


def test_cifar_graph_compiles_and_matches_demo_config(
    cifar_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    result = compile_workflow(cifar_workflow, _options(tmp_path))
    assert result.ok, [issue.as_dict() for issue in result.issues]
    assert result.config_path
    generated = _resolved_config(Path(result.config_path))
    demo = _resolved_config(DEMO_CONFIG)

    for path in (
        ("model", "target"),
        ("model", "params", "num_classes"),
        ("model", "params", "dropout"),
        ("trainer", "max_epochs"),
        ("trainer", "batch_size"),
        ("trainer", "grad_clip_max_norm"),
        ("optimization", "optimizer", "target"),
        ("optimization", "optimizer", "params", "lr"),
        ("optimization", "optimizer", "params", "momentum"),
        ("optimization", "optimizer", "params", "weight_decay"),
        ("optimization", "scheduler", "target"),
        ("optimization", "scheduler", "params", "step_size"),
        ("train_dataset", "target"),
        ("train_dataset", "params", "size"),
        ("train_dataset", "params", "num_classes"),
        ("train_dataset", "params", "random_offset"),
        ("val_dataset", "params", "size"),
        ("train_dataloader", "params", "shuffle"),
        ("train_dataloader", "params", "drop_last"),
        ("checkpoint", "monitor"),
        ("checkpoint", "mode"),
    ):
        assert _dig(generated, path) == _dig(demo, path), path

    # 模板把 num_workers 放在 dataloader_defaults, 生成配置放在每个 loader 上;
    # 两者表达同一语义, 因此比较生效值而不是字段位置.
    assert _dig(generated, ("train_dataloader", "params", "num_workers")) == _dig(
        demo, ("dataloader_defaults", "num_workers")
    )
    assert _dig(generated, ("loss", 0, "target")) == "torch.nn:CrossEntropyLoss"
    assert _dig(generated, ("metrics", 0, "target")) == "registry:Accuracy"
    assert [
        item["target"]
        for item in _dig(generated, ("train_transforms", "params", "transforms"))
    ] == [
        item["target"]
        for item in _dig(demo, ("train_transforms", "params", "transforms"))
    ]
    assert _dig(generated, ("train_dataset", "params", "transform")) == _dig(
        demo, ("train_dataset", "params", "transform")
    )


def test_generated_config_builds_through_setup_from_yaml(
    cifar_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    result = compile_workflow(cifar_workflow, _options(tmp_path))
    assert result.ok, [issue.as_dict() for issue in result.issues]
    setup = setup_from_yaml(result.config_path)
    assert setup.train_loader.batch_size == 64
    assert len(setup.train_loader.dataset) == 512
    assert len(setup.val_loader.dataset) == 128
    assert type(setup.optimizer).__name__ == "SGD"
    assert setup.trainer.max_epochs == 1
    assert [type(metric).__name__ for metric in setup.metrics] == ["Accuracy"]


def test_node_edits_reach_generated_config(
    cifar_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    payload = copy.deepcopy(cifar_workflow)
    _node(payload, "model")["target"] = "registry:resnet18"
    _node(payload, "model")["params"] = {"num_classes": 10}
    _node(payload, "trainer")["params"]["batch_size"] = 32
    _node(payload, "trainer")["params"]["max_epochs"] = 4
    _node(payload, "opt")["params"]["lr"] = 0.05

    result = compile_workflow(payload, _options(tmp_path))
    assert result.ok, [issue.as_dict() for issue in result.issues]
    generated = _resolved_config(Path(result.config_path))
    assert _dig(generated, ("model", "target")) == "registry:resnet18"
    assert _dig(generated, ("trainer", "batch_size")) == 32
    assert _dig(generated, ("trainer", "max_epochs")) == 4
    assert _dig(generated, ("optimization", "optimizer", "params", "lr")) == 0.05


def test_unknown_target_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    _node(payload, "model")["target"] = "registry:not_registered"
    result = compile_workflow(payload, _options())
    assert not result.ok
    assert "unknown_target" in {issue.code for issue in result.issues}
    assert result.config is None


def test_schema_v1_profile_rejects_dag_only_nodes(
    cpu_workflow: Dict[str, Any],
) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["nodes"].append(
        {"id": "ident", "kind": "identity", "target": "builtin:Identity", "params": {}}
    )
    payload["edges"].append(
        {
            "id": "e-ident",
            "from": {"node": "model", "port": "module"},
            "to": {"node": "ident", "port": "in"},
        }
    )
    payload["edges"].append(
        {
            "id": "e-ident-out",
            "from": {"node": "ident", "port": "out"},
            "to": {"node": "opt", "port": "model"},
        }
    )
    result = compile_workflow(payload, _options())
    assert not result.ok
    assert "unsupported_profile" in {issue.code for issue in result.issues}


def test_validate_only_does_not_write_files(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    result = validate_only(
        cpu_workflow,
        CompileOptions(
            data_root=str(REPO_ROOT / "data"),
            project_root=str(REPO_ROOT),
            output_path=str(tmp_path / "config.yaml"),
            system_overrides=dict(SYSTEM_OVERRIDES),
        ),
    )
    assert result.ok
    assert not (tmp_path / "config.yaml").exists()


def test_oversized_params_are_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    _node(payload, "trainer")["params"]["max_epochs"] = 10**12
    result = compile_workflow(payload, _options())
    assert not result.ok
    assert "invalid_param" in {issue.code for issue in result.issues}


def test_dag_runtime_compiles_without_yaml(dag_workflow: Dict[str, Any]) -> None:
    result = compile_workflow(dag_workflow, _options())
    assert result.ok, [issue.as_dict() for issue in result.issues]
    assert result.execution_profile == "dag-runtime"
    assert result.config is None
    assert result.config_path == ""
    assert result.runtime_manifest["profile"] == "dag-runtime"
    assert result.runtime_manifest["topological_order"][-1] == "trainer"


def test_data_path_param_is_resolved_under_data_root(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "name": "cifar-path",
        "execution_profile": "schema-v1",
        "nodes": [
            {
                "id": "ds",
                "kind": "dataset",
                "target": "torchvision.datasets:CIFAR10",
                "params": {"root": "cifar10", "download": False},
            },
            {
                "id": "dl",
                "kind": "dataloader",
                "target": "builtin:DataLoader",
                "params": {"shuffle": True},
            },
            {
                "id": "model",
                "kind": "model",
                "target": "registry:simple_mlp",
                "params": {"input_size": 3072},
            },
            {
                "id": "loss",
                "kind": "loss",
                "target": "torch.nn:CrossEntropyLoss",
                "params": {},
            },
            {
                "id": "opt",
                "kind": "optimizer",
                "target": "torch.optim:Adam",
                "params": {},
            },
            {
                "id": "trainer",
                "kind": "trainer",
                "target": "builtin:Trainer",
                "params": {"batch_size": 8},
            },
        ],
        "edges": [
            {
                "id": "e1",
                "from": {"node": "ds", "port": "samples"},
                "to": {"node": "dl", "port": "dataset"},
            },
            {
                "id": "e2",
                "from": {"node": "dl", "port": "batches"},
                "to": {"node": "trainer", "port": "train_loader"},
            },
            {
                "id": "e3",
                "from": {"node": "model", "port": "module"},
                "to": {"node": "trainer", "port": "model"},
            },
            {
                "id": "e4",
                "from": {"node": "loss", "port": "loss"},
                "to": {"node": "trainer", "port": "loss"},
            },
            {
                "id": "e5",
                "from": {"node": "opt", "port": "optimizer"},
                "to": {"node": "trainer", "port": "optimizer"},
            },
            {
                "id": "e6",
                "from": {"node": "model", "port": "module"},
                "to": {"node": "opt", "port": "model"},
            },
        ],
    }
    data_root = tmp_path / "data"
    data_root.mkdir()
    result = compile_workflow(
        payload,
        CompileOptions(
            data_root=str(data_root),
            project_root=str(REPO_ROOT),
            system_overrides=dict(SYSTEM_OVERRIDES),
            check_registry=False,
        ),
    )
    assert result.ok, [issue.as_dict() for issue in result.issues]
    assert _dig(result.config, ("train_dataset", "params", "root")) == str(
        data_root / "cifar10"
    )


def test_multiple_loss_nodes_lower_to_weighted_list(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["nodes"].append(
        {
            "id": "loss-2",
            "kind": "loss",
            "target": "torch.nn:MSELoss",
            "params": {},
        }
    )
    payload["edges"].append(
        {
            "id": "e-loss-2",
            "from": {"node": "loss-2", "port": "loss"},
            "to": {"node": "trainer", "port": "loss"},
        }
    )
    result = compile_workflow(payload, _options(tmp_path))
    assert result.ok, [issue.as_dict() for issue in result.issues]
    assert [item["target"] for item in result.config["loss"]] == [
        "torch.nn:CrossEntropyLoss",
        "torch.nn:MSELoss",
    ]


def _dig(payload: Any, path: tuple) -> Any:
    current = payload
    for key in path:
        current = current[key]
    return current
