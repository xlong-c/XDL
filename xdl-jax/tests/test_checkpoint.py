"""checkpoint 和 exact resume 测试."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from xdl_jax import JaxCheckpointManager, JaxTrainer, TrainerConfig
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData


def _train(max_epochs: int, data: SyntheticRegressionData):
    return JaxTrainer(
        LinearRegressionTask(input_dim=2, learning_rate=0.04),
        config=TrainerConfig(max_epochs=max_epochs),
    ).fit(data)


def test_checkpoint_exact_restore(tmp_path: Path) -> None:
    data = SyntheticRegressionData(
        n_samples=32,
        input_dim=2,
        batch_size=8,
        seed=12,
    )
    first = _train(1, data)
    with JaxCheckpointManager(tmp_path / "ckpt", asynchronous=False) as manager:
        report = manager.save(
            first.state.optimizer_step,
            first.state,
            data_state=data.state_dict(),
            metadata={"test": "exact"},
        )
        assert report.saved
        restored = manager.restore(target=first.state)
        assert restored.metadata["test"] == "exact"
        assert restored.data_state["exact_iterator_resume"] is False
        assert restored.state.micro_step == first.state.micro_step
        assert restored.state.optimizer_step == first.state.optimizer_step


def test_checkpoint_weights_only_keeps_target_runtime_state(tmp_path: Path) -> None:
    data = SyntheticRegressionData(
        n_samples=16,
        input_dim=2,
        batch_size=8,
        seed=3,
    )
    result = _train(1, data)
    with JaxCheckpointManager(tmp_path / "ckpt", asynchronous=False) as manager:
        manager.save(1, result.state, metadata={"mode": "weights"})
        restored = manager.restore(
            target=result.state.replace(optimizer_step=99),
            mode="weights_only",
        )
    assert restored.state.optimizer_step == 99
    assert np.allclose(
        np.asarray(restored.state.model_state.params["weight"]),
        np.asarray(result.state.model_state.params["weight"]),
    )


def test_interrupted_and_restored_training_matches_continuous_run(
    tmp_path: Path,
) -> None:
    continuous_data = SyntheticRegressionData(
        n_samples=32,
        input_dim=2,
        batch_size=8,
        seed=21,
    )
    continuous = _train(2, continuous_data)

    interrupted_data = SyntheticRegressionData(
        n_samples=32,
        input_dim=2,
        batch_size=8,
        seed=21,
    )
    first = _train(1, interrupted_data)
    with JaxCheckpointManager(tmp_path / "resume", asynchronous=False) as manager:
        manager.save(
            first.state.optimizer_step,
            first.state,
            data_state=interrupted_data.state_dict(),
        )
        restored = manager.restore(target=first.state)

    resumed = JaxTrainer(
        LinearRegressionTask(input_dim=2, learning_rate=0.04),
        config=TrainerConfig(max_epochs=2),
    ).fit(interrupted_data, state=restored.state)

    continuous_leaves = [
        np.asarray(value)
        for value in jax.tree_util.tree_leaves(
            continuous.state.model_state.params
        )
    ]
    resumed_leaves = [
        np.asarray(value)
        for value in jax.tree_util.tree_leaves(
            resumed.state.model_state.params
        )
    ]
    assert len(continuous_leaves) == len(resumed_leaves)
    assert all(
        np.allclose(left, right, atol=1e-6)
        for left, right in zip(continuous_leaves, resumed_leaves, strict=True)
    )
    assert resumed.state.rng_key.tolist() == continuous.state.rng_key.tolist()
