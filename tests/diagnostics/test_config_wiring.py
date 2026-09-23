"""Config-driven diagnostics wiring tests.

Verifies that ``diagnostics.enabled`` in a schema-v1 YAML causes
``Trainer.from_setup`` to attach exactly one ``DiagnosticsCallback`` and that
the disabled path attaches none. Uses the repository's real example YAML so
path resolvers (``${xdl.config_dir}``) stay valid.
"""

from __future__ import annotations

import yaml
from pathlib import Path

import pytest

from xdl.callbacks.diagnostics_callback import DiagnosticsCallback
from xdl.config import setup_from_yaml
from xdl.trainer.trainer import Trainer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _REPO_ROOT / "config" / "manifest_regression_example.yaml"


def _write_variant(tmp_path: Path, enabled: bool) -> Path:
    """Copy the example config, pinning manifest paths to absolute locations.

    ``${xdl.config_dir}`` resolves relative to the YAML file, so a variant
    written outside ``config/`` must not rely on it.
    """
    config = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))
    config["diagnostics"] = {"enabled": enabled, "deep_dive": False}
    assets = _REPO_ROOT / "examples" / "assets" / "manifest_regression"
    for split in ("train", "val"):
        config[f"{split}_dataset"]["params"]["manifest_path"] = str(
            assets / f"{split}.jsonl"
        )
    target = tmp_path / ("enabled.yaml" if enabled else "disabled.yaml")
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    return target


def _diagnostics_callbacks(trainer: Trainer) -> list:
    return [
        cb
        for cb in trainer.callback_list.callbacks
        if isinstance(cb, DiagnosticsCallback)
    ]


@pytest.mark.skipif(not _EXAMPLE.exists(), reason="example config not present")
def test_enabled_attaches_one_diagnostics_callback(tmp_path: Path) -> None:
    trainer = Trainer.from_setup(setup_from_yaml(_write_variant(tmp_path, True)))
    callbacks = _diagnostics_callbacks(trainer)
    assert len(callbacks) == 1


@pytest.mark.skipif(not _EXAMPLE.exists(), reason="example config not present")
def test_disabled_attaches_no_diagnostics_callback(tmp_path: Path) -> None:
    trainer = Trainer.from_setup(setup_from_yaml(_write_variant(tmp_path, False)))
    assert _diagnostics_callbacks(trainer) == []


def test_diagnostics_config_defaults_disabled() -> None:
    from xdl.config.schema import DiagnosticsConfig

    cfg = DiagnosticsConfig()
    assert cfg.enabled is False
    assert cfg.deep_dive is True
    assert cfg.max_deep_dives == 1
