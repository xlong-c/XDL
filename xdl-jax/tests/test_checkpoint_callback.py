"""checkpoint callback 集成测试."""

from __future__ import annotations

from pathlib import Path

from xdl_jax import (
    CheckpointCallback,
    JaxCheckpointManager,
    JaxTrainer,
    TrainerConfig,
)
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData


def test_checkpoint_callback_saves_epoch_state(tmp_path: Path) -> None:
    data = SyntheticRegressionData(
        n_samples=16,
        input_dim=2,
        batch_size=4,
        seed=31,
    )
    callback = CheckpointCallback(
        str(tmp_path / "callback"),
        asynchronous=False,
    )
    result = JaxTrainer(
        LinearRegressionTask(input_dim=2),
        config=TrainerConfig(max_epochs=2),
        callbacks=[callback],
    ).fit(data)

    with JaxCheckpointManager(tmp_path / "callback", asynchronous=False) as manager:
        assert manager.latest_step == result.state.optimizer_step
        restored = manager.restore(target=result.state)
    assert restored.metadata["format_version"] == 1
    assert restored.data_state["exact_iterator_resume"] is False
