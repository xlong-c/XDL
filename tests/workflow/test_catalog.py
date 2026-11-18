"""Node catalog tests."""

from __future__ import annotations

import pytest

from xdl.workflow import load_catalog
from xdl.workflow.catalog import (
    PARAM_TYPES,
    apply_param_defaults,
    check_kind_and_target,
    validate_params,
)
from xdl.workflow.errors import Issue


def test_catalog_declares_every_referenced_kind() -> None:
    catalog = load_catalog()
    assert catalog.version == 1
    for kind in ("dataset", "transform", "dataloader", "model", "loss"):
        assert catalog.kind(kind) is not None
    for kind in ("identity", "compose", "merge", "sink"):
        assert catalog.profiles_of(kind) == ("dag-runtime",)


def test_catalog_targets_use_supported_prefixes() -> None:
    catalog = load_catalog()
    for kind_spec in catalog.kinds.values():
        assert kind_spec.targets, kind_spec.kind
        for target in kind_spec.targets:
            source = target.target.split(":", 1)[0]
            assert source in {"registry", "builtin"} or source.startswith(
                ("torch.", "torchvision.")
            ), target


def test_catalog_param_types_are_closed() -> None:
    catalog = load_catalog()
    for kind_spec in catalog.kinds.values():
        for target in kind_spec.targets:
            for param in target.params:
                assert param.type in PARAM_TYPES, (kind_spec.kind, param.name)


def test_catalog_limits_are_positive() -> None:
    limits = load_catalog().limits
    assert limits.max_nodes > 0
    assert limits.max_edges > 0
    assert limits.max_params_depth > 0
    assert limits.max_params_bytes > 0
    assert limits.max_snapshot_bytes > 0


def test_check_kind_and_target_reports_unknown_kind() -> None:
    catalog = load_catalog()
    _, _, issues = check_kind_and_target(
        catalog, node_id="n1", kind="ghost", target="registry:vgg11_bn"
    )
    assert [issue.code for issue in issues] == ["unknown_node_kind"]


def test_check_kind_and_target_reports_unknown_target() -> None:
    catalog = load_catalog()
    _, target_spec, issues = check_kind_and_target(
        catalog, node_id="n1", kind="model", target="registry:ghost"
    )
    assert target_spec is None
    assert [issue.code for issue in issues] == ["unknown_target"]


def test_validate_params_rejects_unknown_name() -> None:
    catalog = load_catalog()
    target = catalog.target("model", "registry:vgg11_bn")
    assert target is not None
    issues = validate_params(target, {"bogus": 1}, node_id="model")
    assert [issue.code for issue in issues] == ["invalid_param"]


def test_validate_params_checks_range_and_type() -> None:
    catalog = load_catalog()
    target = catalog.target("trainer", "builtin:Trainer")
    assert target is not None
    assert validate_params(target, {"max_epochs": 1}, node_id="t") == []
    issues = validate_params(target, {"max_epochs": 0}, node_id="t")
    assert [issue.code for issue in issues] == ["invalid_param"]
    issues = validate_params(target, {"max_epochs": True}, node_id="t")
    assert [issue.code for issue in issues] == ["invalid_param"]
    issues = validate_params(target, {"precision": "float64"}, node_id="t")
    assert [issue.code for issue in issues] == ["invalid_param"]


def test_validate_params_requires_required_field() -> None:
    catalog = load_catalog()
    target = catalog.target("transform", "torchvision.transforms:Normalize")
    assert target is not None
    issues = validate_params(target, {"mean": [0.5]}, node_id="tf")
    assert [issue.code for issue in issues] == ["invalid_param"]


def test_validate_params_rejects_traversal_path() -> None:
    catalog = load_catalog()
    target = catalog.target("dataset", "torchvision.datasets:CIFAR10")
    assert target is not None
    for value in ("/abs/path", "../up"):
        issues = validate_params(target, {"root": value}, node_id="ds")
        assert [issue.code for issue in issues] == ["path_not_allowed"]


def test_apply_param_defaults_fills_missing_values() -> None:
    catalog = load_catalog()
    target = catalog.target("trainer", "builtin:Trainer")
    assert target is not None
    resolved = apply_param_defaults(target, {"max_epochs": 3})
    assert resolved["max_epochs"] == 3
    assert resolved["batch_size"] == 64
    assert resolved["precision"] == "32"


def test_issue_rejects_unknown_code() -> None:
    with pytest.raises(ValueError):
        Issue(code="not_a_code", message="x")
