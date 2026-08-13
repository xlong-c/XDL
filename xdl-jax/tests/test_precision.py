"""precision 配置测试."""

from __future__ import annotations

import numpy as np

from xdl_jax import JaxTrainer, TrainerConfig
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData


def test_bfloat16_cast_keeps_training_running() -> None:
    data = SyntheticRegressionData(
        n_samples=16,
        input_dim=2,
        batch_size=4,
        seed=41,
    )
    result = JaxTrainer(
        LinearRegressionTask(input_dim=2),
        config=TrainerConfig(max_epochs=1, precision="bfloat16"),
    ).fit(data)
    assert result.history
    assert np.isfinite(result.history[-1].loss)
