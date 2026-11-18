"""dag-runtime executor tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch

from xdl.callbacks import DashboardCallback
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer, TrainerControl
from xdl.workflow import WorkflowSnapshot, WorkflowValidationError, parse_snapshot
from xdl.workflow.executor import SUPPORTED_KINDS, WorkflowExecutor

from dashboard.worker.workflow_compiler import CompileOptions, compile_workflow

REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_OVERRIDES = {
    "runtime.device": "cpu",
    "logging.enable_console": False,
    "logging.enable_tensorboard": False,
    "logging.enable_wandb": False,
}


def _overrides(tmp_path: Path) -> Dict[str, Any]:
    return {
        **SYSTEM_OVERRIDES,
        "runtime.output_dir": str(tmp_path),
        "logging.log_dir": str(tmp_path / "logs"),
        "checkpoint.dirpath": str(tmp_path / "checkpoints"),
    }


def _as_dag_runtime(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(payload)
    result["execution_profile"] = "dag-runtime"
    return result


def _node(payload: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    for node in payload["nodes"]:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"node {node_id} not found")


def test_supported_kinds_cover_catalog() -> None:
    from xdl.workflow import load_catalog

    catalog = load_catalog()
    assert set(catalog.kind_names()) == SUPPORTED_KINDS


def test_linear_graph_builds_setup(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    payload = _as_dag_runtime(cpu_workflow)
    executor = WorkflowExecutor()
    setup = executor.build_setup(
        parse_snapshot(payload), system_overrides=_overrides(tmp_path)
    )
    assert type(setup.model).__name__ == "SimpleMLP"
    assert setup.train_loader.batch_size == 16
    assert len(setup.train_loader.dataset) == 64
    assert type(setup.optimizer).__name__ == "Adam"
    assert type(setup.loss_fn).__name__ == "CrossEntropyLoss"
    assert setup.trainer.max_epochs == 1
    assert setup.runtime.device == "cpu"
    assert setup.runtime.seed == 42
    assert setup.checkpoint.dirpath == str(tmp_path / "checkpoints")


def test_merge_node_concatenates_datasets(
    dag_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    executor = WorkflowExecutor()
    result = executor.execute(
        parse_snapshot(dag_workflow), system_overrides=_overrides(tmp_path)
    )
    dataset = result.setup.train_loader.dataset
    assert type(dataset).__name__ == "ConcatDataset"
    assert len(dataset) == 64
    assert result.order[-1] == "trainer"


def test_compose_node_merges_transform_branches(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "name": "compose branch",
        "execution_profile": "dag-runtime",
        "nodes": [
            {
                "id": "ds",
                "kind": "dataset",
                "target": "torchvision.datasets:FakeData",
                "params": {
                    "size": 16,
                    "image_size": [3, 8, 8],
                    "num_classes": 2,
                },
            },
            {
                "id": "tf-a",
                "kind": "transform",
                "target": "torchvision.transforms:Normalize",
                "params": {"mean": [0.5], "std": [0.5]},
            },
            {
                "id": "tf-b",
                "kind": "transform",
                "target": "torchvision.transforms:RandomHorizontalFlip",
                "params": {"p": 0.0},
            },
            {
                "id": "compose",
                "kind": "compose",
                "target": "builtin:Compose",
                "params": {},
            },
            {
                "id": "dl",
                "kind": "dataloader",
                "target": "builtin:DataLoader",
                "params": {"shuffle": False},
            },
            {
                "id": "model",
                "kind": "model",
                "target": "registry:simple_mlp",
                "params": {"input_size": 192, "num_classes": 2},
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
                "params": {"batch_size": 4},
            },
        ],
        "edges": [
            {
                "id": "e1",
                "from": {"node": "tf-a", "port": "pipeline"},
                "to": {"node": "compose", "port": "left"},
            },
            {
                "id": "e2",
                "from": {"node": "tf-b", "port": "pipeline"},
                "to": {"node": "compose", "port": "right"},
            },
            {
                "id": "e3",
                "from": {"node": "compose", "port": "out"},
                "to": {"node": "ds", "port": "transform"},
            },
            {
                "id": "e4",
                "from": {"node": "ds", "port": "samples"},
                "to": {"node": "dl", "port": "dataset"},
            },
            {
                "id": "e5",
                "from": {"node": "dl", "port": "batches"},
                "to": {"node": "trainer", "port": "train_loader"},
            },
            {
                "id": "e6",
                "from": {"node": "model", "port": "module"},
                "to": {"node": "trainer", "port": "model"},
            },
            {
                "id": "e7",
                "from": {"node": "loss", "port": "loss"},
                "to": {"node": "trainer", "port": "loss"},
            },
            {
                "id": "e8",
                "from": {"node": "opt", "port": "optimizer"},
                "to": {"node": "trainer", "port": "optimizer"},
            },
            {
                "id": "e9",
                "from": {"node": "model", "port": "module"},
                "to": {"node": "opt", "port": "model"},
            },
        ],
    }
    setup = WorkflowExecutor().build_setup(
        parse_snapshot(payload), system_overrides=_overrides(tmp_path)
    )
    transform = setup.train_loader.dataset.transform
    assert type(transform).__name__ == "Compose"
    assert [type(op).__name__ for op in transform.transforms] == [
        "Normalize",
        "RandomHorizontalFlip",
    ]


def test_multiple_loss_nodes_build_weighted_loss(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    payload = _as_dag_runtime(cpu_workflow)
    payload["nodes"].append(
        {"id": "loss-2", "kind": "loss", "target": "torch.nn:MSELoss", "params": {}}
    )
    payload["edges"].append(
        {
            "id": "e-loss-2",
            "from": {"node": "loss-2", "port": "loss"},
            "to": {"node": "trainer", "port": "loss"},
        }
    )
    setup = WorkflowExecutor().build_setup(
        parse_snapshot(payload), system_overrides=_overrides(tmp_path)
    )
    assert type(setup.loss_fn).__name__ == "WeightedLoss"
    assert len(setup.loss_fn.losses) == 2


def test_cycle_is_rejected_before_execution(cpu_workflow: Dict[str, Any]) -> None:
    payload = _as_dag_runtime(cpu_workflow)
    payload["edges"].append(
        {
            "id": "e-cycle",
            "from": {"node": "trainer", "port": "run"},
            "to": {"node": "ds", "port": "transform"},
        }
    )
    with pytest.raises(WorkflowValidationError):
        parse_snapshot(payload)


def test_unsupported_kind_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    snapshot = parse_snapshot(_as_dag_runtime(cpu_workflow))
    patched = WorkflowSnapshot(
        version=snapshot.version,
        id=snapshot.id,
        name=snapshot.name,
        execution_profile=snapshot.execution_profile,
        nodes=tuple(
            node.__class__(
                id=node.id,
                kind="teleporter" if node.kind == "model" else node.kind,
                target=node.target,
                params=node.params,
                position=node.position,
                metadata=node.metadata,
            )
            for node in snapshot.nodes
        ),
        edges=snapshot.edges,
        settings=snapshot.settings,
    )
    with pytest.raises(WorkflowValidationError):
        WorkflowExecutor().build_setup(patched)


def test_runtime_setup_trains_and_emits_dashboard_events(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    payload = _as_dag_runtime(cpu_workflow)
    setup = WorkflowExecutor().build_setup(
        parse_snapshot(payload), system_overrides=_overrides(tmp_path)
    )
    events: List[Tuple[str, Dict[str, Any], bool]] = []
    setup.callbacks.append(
        DashboardCallback(
            lambda event_type, event_payload, reliable: events.append(
                (event_type, event_payload, reliable)
            ),
            every_n_steps=1,
            interval_s=0.0,
        )
    )
    trainer = Trainer.from_setup(setup)
    model = setup.create_model()
    trainer.fit(model, setup.train_loader, setup.val_loader)

    assert trainer.global_step > 0
    event_types = {item[0] for item in events}
    assert "run_state" in event_types
    assert "metrics" in event_types
    checkpoint_dir = tmp_path / "checkpoints"
    assert checkpoint_dir.exists()
    assert any(checkpoint_dir.iterdir())


def test_runtime_setup_honors_stop_control(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    payload = _as_dag_runtime(cpu_workflow)
    payload["nodes"] = [
        node
        if node["id"] != "trainer"
        else {**node, "params": {"batch_size": 8, "max_epochs": 5}}
        for node in payload["nodes"]
    ]
    setup = WorkflowExecutor().build_setup(
        parse_snapshot(payload), system_overrides=_overrides(tmp_path)
    )
    control = TrainerControl()
    trainer = Trainer.from_setup(setup)
    model = setup.create_model()

    class StopAfterTwo:
        def on_train_batch_end(
            self, trainer: Trainer, core_module: Any, **kwargs: Any
        ) -> None:
            del core_module, kwargs
            if trainer.global_step == 2:
                control.request_stop(save_checkpoint=True)

    trainer.callback_list.add_callback(StopAfterTwo())
    trainer.fit(model, setup.train_loader, control=control)
    assert trainer.global_step == 2
    assert control.stop_mode == "stop"


def test_pause_and_resume_handshake_works_on_runtime_setup(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    import threading

    setup = WorkflowExecutor().build_setup(
        parse_snapshot(_as_dag_runtime(cpu_workflow)),
        system_overrides=_overrides(tmp_path),
    )
    control = TrainerControl()
    control.request_pause()
    resumed = threading.Event()

    def wait_for_resume() -> None:
        control.wait_if_paused()
        resumed.set()

    thread = threading.Thread(target=wait_for_resume)
    thread.start()
    assert not resumed.wait(timeout=0.05)
    control.resume()
    thread.join(timeout=2)
    assert resumed.is_set()
    del setup


def test_schema_v1_and_dag_runtime_agree_on_standard_graph(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    compiled = compile_workflow(
        cpu_workflow,
        CompileOptions(
            data_root=str(REPO_ROOT / "data"),
            project_root=str(REPO_ROOT),
            output_path=str(tmp_path / "config.yaml"),
            system_overrides=_overrides(tmp_path),
        ),
    )
    assert compiled.ok, [issue.as_dict() for issue in compiled.issues]
    schema_setup = setup_from_yaml(compiled.config_path)
    runtime_setup = WorkflowExecutor().build_setup(
        parse_snapshot(_as_dag_runtime(cpu_workflow)),
        system_overrides=_overrides(tmp_path),
    )

    assert type(schema_setup.model).__name__ == type(runtime_setup.model).__name__
    assert (
        type(schema_setup.optimizer).__name__ == type(runtime_setup.optimizer).__name__
    )
    assert (
        schema_setup.optimizer.defaults["lr"] == runtime_setup.optimizer.defaults["lr"]
    )
    assert type(schema_setup.loss_fn).__name__ == type(runtime_setup.loss_fn).__name__
    assert schema_setup.train_loader.batch_size == runtime_setup.train_loader.batch_size
    assert len(schema_setup.train_loader.dataset) == len(
        runtime_setup.train_loader.dataset
    )
    assert schema_setup.trainer.max_epochs == runtime_setup.trainer.max_epochs
    assert [type(item).__name__ for item in schema_setup.metrics] == [
        type(item).__name__ for item in runtime_setup.metrics
    ]


def test_runtime_setup_produces_trainable_batch(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    setup = WorkflowExecutor().build_setup(
        parse_snapshot(_as_dag_runtime(cpu_workflow)),
        system_overrides=_overrides(tmp_path),
    )
    batch = next(iter(setup.train_loader))
    assert isinstance(batch, (list, tuple))
    assert torch.is_tensor(batch[0])
    assert batch[0].shape[0] == setup.train_loader.batch_size
