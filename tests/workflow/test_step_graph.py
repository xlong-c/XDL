"""The optional batch graph drives real Trainer steps."""

from pathlib import Path
from typing import Any

import pytest
import torch

from dashboard.worker.workflow_validator import validate_workflow
from xdl.trainer import Trainer
from xdl.workflow import WorkflowExecutionError, load_catalog, parse_snapshot
from xdl.workflow.executor import WorkflowExecutor
from xdl.workflow.step_graph import compile_step_graph


@pytest.fixture
def step_payload(cpu_workflow: dict[str, Any]) -> dict[str, Any]:
    cpu_workflow["execution_profile"] = "dag-runtime"
    nodes = [
        ("batch", "BatchInput", {}),
        ("x", "Select", {"index": 0}),
        ("y", "Select", {"index": 1}),
        ("prediction", "Forward", {}),
        ("loss", "LossCall", {"component": "loss"}),
        ("total", "WeightedSum", {"left_weight": 1.5, "right_weight": 0.5}),
        ("output", "StepOutput", {}),
    ]
    connections = [
        ("batch", "batch", "x", "batch"),
        ("batch", "batch", "y", "batch"),
        ("x", "value", "prediction", "features"),
        ("prediction", "predictions", "loss", "predictions"),
        ("y", "value", "loss", "labels"),
        ("loss", "loss", "total", "left"),
        ("loss", "loss", "total", "right"),
        ("total", "loss", "output", "loss"),
    ]
    cpu_workflow["step_graph"] = {
        "nodes": [
            {"id": node, "kind": kind, "target": f"builtin:{kind}", "params": params}
            for node, kind, params in nodes
        ],
        "edges": [
            {"from": {"node": src, "port": out}, "to": {"node": dst, "port": inp}}
            for src, out, dst, inp in connections
        ],
    }
    return cpu_workflow


@pytest.mark.parametrize("dictionary", [False, True])
def test_evaluates_a_weighted_loss(
    step_payload: dict[str, Any], dictionary: bool
) -> None:
    features = torch.randn(4, 4)
    labels = torch.tensor([0, 1, 0, 1])
    batch: Any = (features, labels)
    if dictionary:
        batch = {"features": features, "labels": labels}
        for node in step_payload["step_graph"]["nodes"]:
            if node["kind"] == "Select":
                node["params"] = {"key": "features" if node["id"] == "x" else "labels"}
    snapshot = parse_snapshot(step_payload)
    assert parse_snapshot(snapshot.canonical_payload()).sha256() == snapshot.sha256()
    program = compile_step_graph(snapshot, load_catalog())
    assert program is not None
    model = torch.nn.Linear(4, 2)
    fn = torch.nn.CrossEntropyLoss()
    loss, metrics = program.evaluate(batch, model, {"loss": fn}, {})
    torch.testing.assert_close(loss, 2 * fn(model(features), labels))
    assert metrics == {}
    loss.backward()
    assert model.weight.grad is not None
    with pytest.raises(WorkflowExecutionError, match="step_graph/nodes/x"):
        program.evaluate({}, model, {"loss": fn}, {})
    invalid_labels = (
        {"features": features, "labels": labels.float()}
        if dictionary
        else (features, labels.float())
    )
    with pytest.raises(WorkflowExecutionError, match="step_graph/nodes/loss"):
        program.evaluate(invalid_labels, model, {"loss": fn}, {})
    with pytest.raises(WorkflowExecutionError, match="step_graph/nodes/loss"):
        program.evaluate(
            batch, model, {"loss": torch.nn.CrossEntropyLoss(reduction="none")}, {}
        )


def test_metric_call_runs_outside_autograd(step_payload: dict[str, Any]) -> None:
    step_payload["nodes"].append(
        {"id": "accuracy", "kind": "metric", "target": "registry:Accuracy"}
    )
    graph = step_payload["step_graph"]
    graph["nodes"].append(
        {
            "id": "score",
            "kind": "MetricCall",
            "target": "builtin:MetricCall",
            "params": {"component": "accuracy", "name": "accuracy"},
        }
    )
    graph["edges"].extend(
        {"from": {"node": src, "port": out}, "to": {"node": dst, "port": inp}}
        for src, out, dst, inp in (
            ("prediction", "predictions", "score", "predictions"),
            ("y", "value", "score", "labels"),
            ("score", "value", "output", "metrics"),
        )
    )
    program = compile_step_graph(parse_snapshot(step_payload), load_catalog())
    assert program is not None

    def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        assert not predictions.requires_grad
        return (predictions.argmax(dim=1) == labels).float().mean()

    model = torch.nn.Linear(4, 2)
    features = torch.randn(4, 4)
    labels = model(features).argmax(dim=1)
    loss, metrics = program.evaluate(
        (features, labels),
        model,
        {"loss": torch.nn.CrossEntropyLoss()},
        {"accuracy": accuracy},
    )
    assert metrics["accuracy"].item() == 1.0
    loss.backward()
    assert model.weight.grad is not None


def test_preflight_locates_bad_connection(step_payload: dict[str, Any]) -> None:
    step_payload["step_graph"]["edges"][0]["to"]["port"] = "wrong"
    report = validate_workflow(step_payload)
    assert not report.ok
    assert any(
        issue.code == "unknown_port" and issue.path.startswith("step_graph/")
        for issue in report.issues
    )


def test_step_graph_runs_in_trainer(
    step_payload: dict[str, Any], tmp_path: Path
) -> None:
    snapshot = parse_snapshot(step_payload)
    setup = WorkflowExecutor().build_setup(
        snapshot,
        system_overrides={
            "runtime.device": "cpu",
            "runtime.output_dir": str(tmp_path),
            "logging.log_dir": str(tmp_path / "logs"),
            "logging.enable_console": False,
            "logging.enable_tensorboard": False,
            "checkpoint.dirpath": str(tmp_path / "checkpoints"),
        },
    )
    model = setup.create_model()
    initial = next(model.parameters()).detach().clone()
    trainer = Trainer.from_setup(setup)
    trainer.fit(model, setup.train_loader)
    assert trainer.global_step == 4
    assert not torch.equal(initial, next(model.parameters()).detach())


def test_split_dict_collate_and_optimizer_cadence(
    step_demo: dict[str, Any], tmp_path: Path
) -> None:
    payload = step_demo
    payload["nodes"].append(
        {
            "id": "schedule",
            "kind": "scheduler",
            "target": "torch.optim.lr_scheduler:StepLR",
            "params": {"step_size": 1, "gamma": 0.5},
        }
    )
    payload["edges"].extend(
        [
            {
                "from": {"node": "opt", "port": "optimizer"},
                "to": {"node": "schedule", "port": "optimizer"},
            },
            {
                "from": {"node": "schedule", "port": "scheduler"},
                "to": {"node": "trainer", "port": "scheduler"},
            },
        ]
    )
    assert validate_workflow(payload).ok
    snapshot = parse_snapshot(payload)
    executor = WorkflowExecutor()
    setup = executor.build_setup(
        snapshot,
        system_overrides={
            "runtime.device": "cpu",
            "runtime.output_dir": str(tmp_path),
            "logging.log_dir": str(tmp_path / "logs"),
            "logging.enable_console": False,
            "logging.enable_tensorboard": False,
            "checkpoint.dirpath": str(tmp_path / "checkpoints"),
        },
    )
    assert setup.train_loader.batch_size == 8
    assert len(setup.train_loader.dataset) == 48
    assert len(setup.val_loader.dataset) == 16
    assert set(setup.train_loader.dataset.indices).isdisjoint(
        setup.val_loader.dataset.indices
    )
    repeat = executor.build_setup(snapshot, system_overrides={"runtime.device": "cpu"})
    assert repeat.train_loader.dataset.indices == setup.train_loader.dataset.indices
    fields = next(iter(setup.val_loader))
    assert fields["features"].shape == (8, 4)
    assert fields["labels"].dtype == torch.int64
    trainer = Trainer.from_setup(setup)
    trainer.fit(setup.create_model(), setup.train_loader, setup.val_loader)
    assert trainer.global_step == 6
    assert setup.scheduler.last_epoch == 3
    assert setup.optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.5**3)


@pytest.mark.parametrize("mismatch", ["format", "features", "labels"])
def test_rejects_known_sample_contract_mismatch(
    step_demo: dict[str, Any], mismatch: str
) -> None:
    payload = step_demo
    dataset = next(node for node in payload["nodes"] if node["id"] == "ds")
    dataset["params"].update(
        {
            "format": {"sample_format": "tuple"},
            "features": {"input_shape": [3]},
            "labels": {"num_classes": 3},
        }[mismatch]
    )
    assert any(
        item.code == "type_mismatch" for item in validate_workflow(payload).issues
    )
