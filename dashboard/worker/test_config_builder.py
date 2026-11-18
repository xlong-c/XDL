"""Template config builder validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dashboard.worker.config_builder import build_template_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_template_builder_applies_allowlisted_override_and_schema(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated.yaml"
    built = build_template_config(
        REPO_ROOT / "config/unified_logger_example.yaml",
        {"trainer.max_epochs": 3},
        output,
        allowed_paths=("trainer.max_epochs",),
    )

    assert built == output
    config = OmegaConf.load(output)
    assert config.trainer.max_epochs == 3
    assert config.model.target == "registry:simple_mlp"


def test_template_builder_supports_accelerate_schema(tmp_path: Path) -> None:
    output = tmp_path / "accelerate.yaml"
    build_template_config(
        REPO_ROOT / "config/unified_logger_example.yaml",
        {"trainer.max_epochs": 1},
        output,
        allowed_paths=("trainer.max_epochs",),
        use_accelerate=True,
    )
    config = OmegaConf.load(output)
    assert config.accelerate.cpu is False


def test_template_builder_rejects_unlisted_or_missing_paths(tmp_path: Path) -> None:
    source = REPO_ROOT / "config/unified_logger_example.yaml"
    output = tmp_path / "generated.yaml"

    with pytest.raises(ValueError, match="not allowed"):
        build_template_config(
            source,
            {"trainer.max_epochs": 3},
            output,
            allowed_paths=(),
        )

    with pytest.raises(ValueError, match="does not define"):
        build_template_config(
            source,
            {"trainer.not_a_field": 3},
            output,
            allowed_paths=("trainer.not_a_field",),
        )
