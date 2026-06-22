"""Thin YOLO detection practice entrypoint for the stage workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt.workflows import optimize_model


CONFIG_ENV = "XQT_YOLO_PRACTICE_CONFIG"
DEFAULT_CONFIG_PATH = REPO_ROOT / "xqt" / "recipes" / "yolo_detection_practice.yaml"


def _config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, str(DEFAULT_CONFIG_PATH))).expanduser()


def _metric_text(metrics: dict[str, Any]) -> str:
    if "eval" in metrics:
        values = metrics["eval"].get("metrics", {})
        return ", ".join(f"{key}={value}" for key, value in values.items()) or "eval"
    if "benchmark" in metrics:
        mean_ms = metrics["benchmark"].get("mean_ms")
        return f"mean_ms={mean_ms}"
    if "export" in metrics:
        artifacts = metrics["export"].get("artifacts", [])
        return f"artifacts={len(artifacts)}"
    return ", ".join(metrics.keys()) or "no metrics"


def main() -> int:
    result = optimize_model(_config_path())
    print("XQT stage workflow summary")
    print(f"best_stage: {result.best_stage}")
    for stage in result.stages:
        status = "accepted" if stage.accepted else "rejected"
        print(f"- {stage.name} [{stage.kind}] {status}: {_metric_text(stage.metrics)}")
    if result.artifacts:
        print("artifacts:")
        for name, value in sorted(result.artifacts.items()):
            print(f"- {name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
