"""配置和 callback 测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from xdl_jax import (
    Callback,
    CheckpointCallback,
    ConfigurationError,
    JaxCheckpointManager,
    JaxTrainer,
)
from xdl_jax.config import setup_from_yaml


def test_yaml_setup_uses_jax_namespace(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
config_version: 1
backend: jax
runtime:
  seed: 5
trainer:
  max_epochs: 2
  gradient_accumulation_steps: 2
  jit: false
task:
  target: xdl_jax.examples:LinearRegressionTask
  params:
    input_dim: 2
train_data:
  target: xdl_jax.examples:SyntheticRegressionData
  params:
    n_samples: 16
    input_dim: 2
    batch_size: 4
    seed: 5
optimization:
  optimizer:
    target: optax:adamw
    params:
      learning_rate: 0.02
      weight_decay: 0.0
callbacks:
  - target: xdl_jax.callbacks:TimerCallback
    params: {}
""",
        encoding="utf-8",
    )
    setup = setup_from_yaml(config_path)
    assert setup.trainer_config.max_epochs == 2
    assert setup.trainer_config.gradient_accumulation_steps == 2
    assert setup.optimizer_config["target"] == "optax:adamw"
    assert len(setup.callbacks) == 1
    assert isinstance(setup.callbacks[0], Callback)


def test_yaml_checkpoint_config_builds_callback_and_runs(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "train.yaml"
    checkpoint_dir = tmp_path / "checkpoints"
    config_path.write_text(
        f"""
config_version: 1
backend: jax
runtime:
  seed: 5
  platform: cpu
trainer:
  max_epochs: 1
  jit: false
task:
  target: xdl_jax.examples:LinearRegressionTask
  params:
    input_dim: 2
train_data:
  target: xdl_jax.examples:SyntheticRegressionData
  params:
    n_samples: 8
    input_dim: 2
    batch_size: 4
    seed: 5
checkpoint:
  directory: "{checkpoint_dir.as_posix()}"
  asynchronous: false
""",
        encoding="utf-8",
    )

    setup = setup_from_yaml(config_path)
    assert setup.checkpoint_config["directory"] == checkpoint_dir.as_posix()
    assert any(isinstance(item, CheckpointCallback) for item in setup.callbacks)

    trainer = JaxTrainer.from_setup(setup)
    result = trainer.fit_from_setup(setup)
    with JaxCheckpointManager(checkpoint_dir, asynchronous=False) as manager:
        assert manager.latest_step == result.state.optimizer_step


@pytest.mark.parametrize(
    "content, message",
    [
        (
            "backend: torch\n"
            "task: {target: xdl_jax.examples:LinearRegressionTask}\n"
            "train_data: {target: xdl_jax.examples:SyntheticRegressionData}\n",
            "backend",
        ),
        (
            "backend: jax\n"
            "unknown_field: true\n"
            "task: {target: xdl_jax.examples:LinearRegressionTask}\n"
            "train_data: {target: xdl_jax.examples:SyntheticRegressionData}\n",
            "unknown",
        ),
    ],
)
def test_yaml_setup_rejects_invalid_schema(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        setup_from_yaml(path)
