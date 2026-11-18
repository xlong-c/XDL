"""Workflow compiler CLI protocol tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from dashboard.worker.workflow_cli import handle_request

from .conftest import REPO_ROOT

SYSTEM_OVERRIDES = {
    "runtime.device": "cpu",
    "logging.enable_console": False,
    "logging.enable_tensorboard": False,
}


def _run_cli(request: Dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dashboard.worker.workflow_cli"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )


def test_validate_action_reports_issues_without_config(
    cpu_workflow: Dict[str, Any],
) -> None:
    response = handle_request(
        {
            "action": "validate",
            "snapshot": cpu_workflow,
            "options": {
                "data_root": str(REPO_ROOT / "data"),
                "check_registry": False,
            },
        }
    )
    assert response["ok"] is True
    assert response["issues"] == []
    assert "config" not in response
    assert response["compiler_version"].startswith("workflow-compiler/")


def test_invalid_snapshot_reports_locatable_issues(
    cpu_workflow: Dict[str, Any],
) -> None:
    payload = json.loads(json.dumps(cpu_workflow))
    payload["nodes"][0]["kind"] = "nope"
    response = handle_request(
        {"action": "validate", "snapshot": payload, "options": {}}
    )
    assert response["ok"] is False
    codes = {issue["code"] for issue in response["issues"]}
    assert "unknown_node_kind" in codes


def test_unknown_action_is_rejected(cpu_workflow: Dict[str, Any]) -> None:
    response = handle_request({"action": "deploy", "snapshot": cpu_workflow})
    assert response["ok"] is False
    assert response["issues"][0]["code"] == "invalid_snapshot"


def test_cli_compile_writes_config_and_keeps_stdout_clean(
    cpu_workflow: Dict[str, Any], tmp_path: Path
) -> None:
    output = tmp_path / "config.yaml"
    completed = _run_cli(
        {
            "action": "compile",
            "snapshot": cpu_workflow,
            "options": {
                "data_root": str(REPO_ROOT / "data"),
                "project_root": str(REPO_ROOT),
                "output_path": str(output),
                "system_overrides": dict(SYSTEM_OVERRIDES),
                "check_registry": True,
            },
        }
    )
    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    assert response["ok"] is True
    assert response["config_path"] == str(output)
    assert len(response["config_hash"]) == 64
    assert output.exists()
    # stdout 必须只有一行协议 JSON
    assert completed.stdout.strip().count("\n") == 0


def test_cli_rejects_malformed_request() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "dashboard.worker.workflow_cli"],
        input="not json",
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert completed.returncode == 2
    response = json.loads(completed.stdout)
    assert response["ok"] is False
    assert response["issues"][0]["code"] == "invalid_snapshot"
