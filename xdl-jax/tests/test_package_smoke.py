"""独立包发布和公共导出 smoke 测试."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import xdl_jax


def test_public_version_and_import_contract() -> None:
    assert xdl_jax.__version__ == "0.1.0"
    assert "JaxTrainer" in xdl_jax.__all__
    assert "torch" not in sys.modules


def test_package_can_import_from_project_root() -> None:
    project_root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import xdl_jax; print(xdl_jax.__version__)",
        ],
        cwd=project_root,
        env={"PYTHONPATH": str(project_root)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "0.1.0"
