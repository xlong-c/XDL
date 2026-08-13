"""真实 GPU 训练 contract 测试."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from xdl_jax import JaxCheckpointManager, JaxTrainer, TrainerConfig
from xdl_jax.distributed import SingleDeviceStrategy
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData


def _gpu_devices() -> tuple[jax.Device, ...]:
    try:
        return tuple(jax.devices("gpu"))
    except RuntimeError:
        return ()


requires_gpu = pytest.mark.skipif(
    not _gpu_devices(),
    reason="CUDA-enabled JAX GPU plugin is required",
)


@requires_gpu
def test_gpu_strategy_selects_cuda_device() -> None:
    strategy = SingleDeviceStrategy(platform="gpu")
    strategy.setup()
    report = strategy.report()
    assert report["platform"] == "gpu"
    assert report["selected_devices"][0].startswith("gpu:")


@requires_gpu
def test_gpu_training_reduces_loss() -> None:
    result = JaxTrainer(
        LinearRegressionTask(input_dim=3, learning_rate=0.05),
        config=TrainerConfig(max_epochs=3, platform="gpu"),
    ).fit(
        SyntheticRegressionData(
            n_samples=64,
            input_dim=3,
            batch_size=8,
            seed=4,
        )
    )
    assert result.history[-1].loss < result.history[0].loss
    assert result.state.model_state.params["weight"].devices()


@requires_gpu
def test_gpu_bfloat16_training_runs() -> None:
    result = JaxTrainer(
        LinearRegressionTask(input_dim=2),
        config=TrainerConfig(
            max_epochs=1,
            precision="bfloat16",
            platform="gpu",
        ),
    ).fit(
        SyntheticRegressionData(
            n_samples=16,
            input_dim=2,
            batch_size=4,
            seed=41,
        )
    )
    assert result.history[-1].loss_finite
    assert result.state.model_state.params["weight"].devices()


@requires_gpu
def test_gpu_checkpoint_round_trip_keeps_cuda_state(tmp_path: Path) -> None:
    result = JaxTrainer(
        LinearRegressionTask(input_dim=2),
        config=TrainerConfig(max_epochs=1, platform="gpu"),
    ).fit(
        SyntheticRegressionData(
            n_samples=16,
            input_dim=2,
            batch_size=4,
            seed=12,
        )
    )
    manager = JaxCheckpointManager(
        tmp_path / "gpu-checkpoint",
        asynchronous=False,
    )
    try:
        manager.save(
            1,
            result.state,
            metadata={"platform": "gpu"},
        )
        restored = manager.restore(
            1,
            target=result.state,
            mode="exact",
        )
    finally:
        manager.close()
    assert restored.step == 1
    assert restored.metadata["platform"] == "gpu"
    assert restored.state.model_state.params["weight"].devices()
