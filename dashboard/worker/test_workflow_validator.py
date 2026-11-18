"""Workflow semantic validation tests."""

from __future__ import annotations

import copy
from typing import Any, Dict

from dashboard.worker.workflow_validator import validate_workflow


def _codes(payload: Dict[str, Any]) -> set[str]:
    report = validate_workflow(payload, check_registry=False)
    return {issue.code for issue in report.issues}


def _node(payload: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    for node in payload["nodes"]:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"node {node_id} not found")


def test_standard_graph_is_valid(cifar_workflow: Dict[str, Any]) -> None:
    report = validate_workflow(cifar_workflow, check_registry=False)
    assert report.ok, [issue.as_dict() for issue in report.issues]
    assert report.execution_profile == "schema-v1"
    assert len(report.nodes) == len(cifar_workflow["nodes"])


def test_unknown_node_kind_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    _node(payload, "dl")["kind"] = "not_a_kind"
    assert "unknown_node_kind" in _codes(payload)


def test_unknown_target_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    _node(payload, "model")["target"] = "registry:not_a_model"
    assert "unknown_target" in _codes(payload)


def test_param_type_error_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    _node(payload, "trainer")["params"]["max_epochs"] = "many"
    assert "invalid_param" in _codes(payload)


def test_undeclared_param_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    _node(payload, "opt")["params"]["learning_rate"] = 0.1
    assert "invalid_param" in _codes(payload)


def test_missing_required_port_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["edges"] = [
        edge
        for edge in payload["edges"]
        if not (edge["to"]["node"] == "trainer" and edge["to"]["port"] == "loss")
    ]
    assert "missing_port" in _codes(payload)


def test_unknown_port_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    for edge in payload["edges"]:
        if edge["id"] == "e-dl":
            edge["to"]["port"] = "train_loaderx"
    assert "unknown_port" in _codes(payload)


def test_type_mismatch_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    for edge in payload["edges"]:
        if edge["id"] == "e-opt-model":
            edge["to"]["port"] = "loss"
            edge["to"]["node"] = "trainer"
        elif edge["id"] == "e-loss":
            edge["from"]["port"] = "loss"
    # model.module (model) -> trainer.loss (loss) 必须被拒绝
    assert "type_mismatch" in _codes(payload)


def test_cycle_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["edges"].append(
        {
            "id": "e-cycle",
            "from": {"node": "trainer", "port": "run"},
            "to": {"node": "model", "port": "module"},
        }
    )
    # model 没有输入端口, 结构层先报 unknown_port; 这里验证不会静默通过
    codes = _codes(payload)
    assert codes & {"unknown_port", "cycle_detected"}


def test_unreachable_node_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["nodes"].append(
        {
            "id": "orphan",
            "kind": "model",
            "target": "registry:simple_mlp",
            "params": {},
        }
    )
    assert "unreachable_node" in _codes(payload)


def test_duplicate_role_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["nodes"].append(
        {
            "id": "opt-2",
            "kind": "optimizer",
            "target": "torch.optim:Adam",
            "params": {},
        }
    )
    payload["edges"].append(
        {
            "id": "e-opt-2",
            "from": {"node": "model", "port": "module"},
            "to": {"node": "opt-2", "port": "model"},
        }
    )
    assert "duplicate_role" in _codes(payload)


def test_profile_mismatch_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
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
    assert "unsupported_profile" in _codes(payload)


def test_absolute_data_path_is_rejected() -> None:
    payload = {
        "version": 1,
        "name": "abs",
        "execution_profile": "schema-v1",
        "nodes": [
            {
                "id": "ds",
                "kind": "dataset",
                "target": "torchvision.datasets:CIFAR10",
                "params": {"root": "/etc/passwd"},
            },
            {
                "id": "dl",
                "kind": "dataloader",
                "target": "builtin:DataLoader",
                "params": {},
            },
            {
                "id": "model",
                "kind": "model",
                "target": "registry:simple_mlp",
                "params": {},
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
                "params": {},
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
    assert "path_not_allowed" in _codes(payload)


def test_traversal_data_path_is_rejected() -> None:
    payload = {
        "version": 1,
        "name": "trav",
        "execution_profile": "schema-v1",
        "nodes": [
            {
                "id": "ds",
                "kind": "dataset",
                "target": "torchvision.datasets:CIFAR10",
                "params": {"root": "../etc"},
            }
        ],
        "edges": [],
    }
    assert "path_not_allowed" in _codes(payload)


def test_dag_runtime_requires_consumed_outputs(
    dag_workflow: Dict[str, Any],
) -> None:
    payload = copy.deepcopy(dag_workflow)
    payload["nodes"].append(
        {
            "id": "ident",
            "kind": "identity",
            "target": "builtin:Identity",
            "params": {},
        }
    )
    payload["edges"].append(
        {
            "id": "e-ident",
            "from": {"node": "model", "port": "module"},
            "to": {"node": "ident", "port": "in"},
        }
    )
    report = validate_workflow(payload, check_registry=False)
    assert "dangling_output" in {issue.code for issue in report.issues}


def test_port_conflict_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["nodes"].append(
        {
            "id": "ds-2",
            "kind": "dataset",
            "target": "registry:SyntheticClassificationDataset",
            "params": {},
        }
    )
    payload["edges"].append(
        {
            "id": "e-ds-2",
            "from": {"node": "ds-2", "port": "samples"},
            "to": {"node": "dl", "port": "dataset"},
        }
    )
    assert "port_conflict" in _codes(payload)


def test_registry_check_reports_unknown_component() -> None:
    payload = {
        "version": 1,
        "name": "registry-miss",
        "execution_profile": "schema-v1",
        "nodes": [
            {
                "id": "model",
                "kind": "model",
                "target": "registry:definitely_not_a_model",
                "params": {},
            }
        ],
        "edges": [],
    }
    report = validate_workflow(payload, check_registry=True)
    assert "unknown_target" in {issue.code for issue in report.issues}
