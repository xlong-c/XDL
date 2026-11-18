"""Shared fixtures for xdl.workflow tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_FIXTURES = REPO_ROOT / "dashboard" / "templates" / "workflows"


def load_workflow(name: str) -> Dict[str, Any]:
    """读取共享工作流 fixture."""
    return json.loads((WORKFLOW_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def cpu_workflow() -> Dict[str, Any]:
    """CPU 冒烟 schema-v1 图."""
    return load_workflow("synthetic-cpu-demo")


@pytest.fixture
def dag_workflow() -> Dict[str, Any]:
    """dag-runtime 分支合并图."""
    return load_workflow("dag-runtime-demo")


@pytest.fixture
def step_demo() -> Dict[str, Any]:
    """包含拆分和字典批次的双层训练图."""
    return load_workflow("step-graph-cpu-demo")


@pytest.fixture
def cifar_workflow() -> Dict[str, Any]:
    """标准 CIFAR-10 图."""
    return load_workflow("cifar10-demo")
