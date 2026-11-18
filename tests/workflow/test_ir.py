"""WorkflowSnapshot IR structure tests."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict

import pytest

from xdl.workflow import (
    EXECUTION_PROFILES,
    SNAPSHOT_VERSION,
    WorkflowValidationError,
    has_errors,
    load_catalog,
    parse_snapshot,
    validate_structure,
)


def _codes(payload: Dict[str, Any]) -> set[str]:
    return {issue.code for issue in validate_structure(payload)}


def _node(payload: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    for node in payload["nodes"]:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"node {node_id} not found")


def test_snapshot_version_and_profiles_are_frozen() -> None:
    assert SNAPSHOT_VERSION == 1
    assert EXECUTION_PROFILES == ("schema-v1", "dag-runtime")


def test_valid_fixture_parses_without_issues(cpu_workflow: Dict[str, Any]) -> None:
    assert validate_structure(cpu_workflow) == []
    snapshot = parse_snapshot(cpu_workflow)
    assert snapshot.version == 1
    assert snapshot.execution_profile == "schema-v1"
    assert snapshot.topological_order()[-1] == "trainer"


def test_missing_version_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload.pop("version")
    assert "unsupported_version" in _codes(payload)


def test_unknown_version_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["version"] = 2
    assert _codes(payload) == {"unsupported_version"}


def test_missing_execution_profile_is_rejected(
    cpu_workflow: Dict[str, Any],
) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload.pop("execution_profile")
    assert "invalid_snapshot" in _codes(payload)


def test_duplicate_node_id_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["nodes"].append(copy.deepcopy(_node(payload, "model")))
    assert "duplicate_id" in _codes(payload)


def test_duplicate_edge_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    duplicate = copy.deepcopy(payload["edges"][0])
    duplicate["id"] = "e-dup"
    payload["edges"].append(duplicate)
    assert "duplicate_edge" in _codes(payload)


def test_self_loop_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["edges"].append(
        {
            "id": "e-self",
            "from": {"node": "dl", "port": "batches"},
            "to": {"node": "dl", "port": "dataset"},
        }
    )
    assert "self_loop" in _codes(payload)


def test_cycle_is_rejected() -> None:
    payload = {
        "version": 1,
        "name": "cycle",
        "execution_profile": "schema-v1",
        "nodes": [
            {"id": "a", "kind": "transform", "target": "t:op", "params": {}},
            {"id": "b", "kind": "transform", "target": "t:op", "params": {}},
        ],
        "edges": [
            {
                "id": "e1",
                "from": {"node": "a", "port": "pipeline"},
                "to": {"node": "b", "port": "pipeline"},
            },
            {
                "id": "e2",
                "from": {"node": "b", "port": "pipeline"},
                "to": {"node": "a", "port": "pipeline"},
            },
        ],
    }
    assert "cycle_detected" in _codes(payload)
    with pytest.raises(WorkflowValidationError):
        parse_snapshot(payload)


def test_unknown_node_reference_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    payload["edges"][0]["to"]["node"] = "ghost"
    assert "unknown_node" in _codes(payload)


def test_node_and_edge_limits_are_enforced(cpu_workflow: Dict[str, Any]) -> None:
    limits = load_catalog().limits
    payload = copy.deepcopy(cpu_workflow)
    payload["nodes"] = [
        {
            "id": f"n{index}",
            "kind": "model",
            "target": "registry:simple_mlp",
            "params": {},
        }
        for index in range(limits.max_nodes + 1)
    ]
    payload["edges"] = []
    assert "limit_exceeded" in _codes(payload)


def test_params_depth_limit_is_enforced(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    nested: Any = 1
    for _ in range(load_catalog().limits.max_params_depth + 1):
        nested = {"a": nested}
    _node(payload, "trainer")["params"] = {"max_epochs": 1, "deep": nested}
    assert "limit_exceeded" in _codes(payload)


def test_params_bytes_limit_is_enforced(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    limits = load_catalog().limits
    _node(payload, "trainer")["params"] = {
        "max_epochs": 1,
        "blob": "x" * (limits.max_params_bytes + 1),
    }
    assert "limit_exceeded" in _codes(payload)


def test_missing_edge_id_is_generated(cpu_workflow: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cpu_workflow)
    for edge in payload["edges"]:
        edge.pop("id", None)
    assert not has_errors(validate_structure(payload))
    snapshot = parse_snapshot(payload)
    assert all(edge.id.startswith("edge-") for edge in snapshot.edges)


def test_canonical_payload_is_stable(cpu_workflow: Dict[str, Any]) -> None:
    snapshot = parse_snapshot(cpu_workflow)
    shuffled = copy.deepcopy(cpu_workflow)
    shuffled["nodes"] = list(reversed(shuffled["nodes"]))
    shuffled["edges"] = list(reversed(shuffled["edges"]))
    assert snapshot.canonical_json() == parse_snapshot(shuffled).canonical_json()
    assert snapshot.sha256() == parse_snapshot(shuffled).sha256()


def test_canonical_json_is_compact(cpu_workflow: Dict[str, Any]) -> None:
    snapshot = parse_snapshot(cpu_workflow)
    text = snapshot.canonical_json()
    assert "\n" not in text
    assert json.loads(text)["version"] == 1
