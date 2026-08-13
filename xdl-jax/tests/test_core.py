"""xdl-jax 核心训练闭环测试."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from xdl_jax import (
    ArrayDataSource,
    DataValidationError,
    JaxTrainer,
    ModelState,
    NNXModelAdapter,
    TrainerConfig,
)
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData


def _tree_allclose(left: object, right: object, atol: float = 1e-6) -> bool:
    def as_array(value: object) -> np.ndarray:
        try:
            return np.asarray(value)
        except TypeError:
            return np.asarray(jax.random.key_data(value))

    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.allclose(as_array(a), as_array(b), atol=atol)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def test_import_does_not_load_torch() -> None:
    package_root = Path(__file__).parents[1]
    script = (
        "import sys; import xdl_jax; "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package_root,
        env={"PYTHONPATH": str(package_root)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_synthetic_regression_loss_decreases() -> None:
    data = SyntheticRegressionData(
        n_samples=64,
        input_dim=3,
        batch_size=8,
        seed=4,
    )
    result = JaxTrainer(
        LinearRegressionTask(input_dim=3, learning_rate=0.05),
        config=TrainerConfig(max_epochs=5),
    ).fit(data)

    assert result.state.optimizer_step == 5 * len(data)
    assert result.history[-1].loss < result.history[0].loss
    assert result.history[-1].loss_finite
    assert result.history[-1].gradients_finite
    assert result.first_compile_time_s is not None


def test_gradient_accumulation_updates_only_at_boundary() -> None:
    data = SyntheticRegressionData(
        n_samples=24,
        input_dim=2,
        batch_size=4,
        seed=9,
    )
    result = JaxTrainer(
        LinearRegressionTask(input_dim=2, learning_rate=0.05),
        config=TrainerConfig(
            max_epochs=1,
            gradient_accumulation_steps=3,
        ),
    ).fit(data)

    assert len(result.history) == len(data)
    assert [item.did_optimizer_step for item in result.history] == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert result.state.optimizer_step == 2
    assert result.state.accumulation_count == 0


def test_incomplete_accumulation_can_be_normalized() -> None:
    data = SyntheticRegressionData(
        n_samples=20,
        input_dim=2,
        batch_size=4,
        seed=7,
    )
    result = JaxTrainer(
        LinearRegressionTask(input_dim=2),
        config=TrainerConfig(
            max_epochs=1,
            gradient_accumulation_steps=3,
            drop_incomplete_accumulation=False,
        ),
    ).fit(data)
    assert result.state.optimizer_step == 2
    assert result.state.accumulation_count == 0


def test_batch_validation_rejects_shape_and_dtype_changes() -> None:
    data = ArrayDataSource(
        {"x": np.ones((8, 2), dtype=np.float32)},
        batch_size=4,
    )
    expected = next(iter(data))
    from xdl_jax.data import BatchSpec, validate_batch

    spec = BatchSpec.from_batch(expected)
    with pytest.raises(DataValidationError, match="shape mismatch"):
        validate_batch({"x": np.ones((4, 3), dtype=np.float32)}, spec)
    with pytest.raises(DataValidationError, match="dtype mismatch"):
        validate_batch({"x": np.ones((4, 2), dtype=np.float64)}, spec)


def test_model_state_is_a_jax_pytree() -> None:
    value = ModelState(
        params={"weight": jnp.ones((2, 1))},
        mutable={"counter": jnp.asarray(0)},
    )
    leaves = jax.tree_util.tree_leaves(value)
    assert len(leaves) == 2
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_nnx_adapter_trains_and_preserves_mutable_state() -> None:
    class TinyNNX(nnx.Module):
        def __init__(self, rngs: nnx.Rngs) -> None:
            self.linear = nnx.Linear(2, 1, rngs=rngs)

        def __call__(self, inputs: jax.Array) -> jax.Array:
            return self.linear(inputs)

    class NNXTask:
        def build_model(self) -> NNXModelAdapter:
            return NNXModelAdapter(lambda rng, batch: TinyNNX(nnx.Rngs(rng)))

        def loss_and_metrics(
            self,
            model: NNXModelAdapter,
            model_state: ModelState,
            batch: dict[str, jax.Array],
            rng: jax.Array,
            *,
            training: bool,
        ) -> tuple[jax.Array, dict[str, jax.Array], ModelState]:
            prediction, updated_state = model.apply(
                model_state,
                batch["x"],
                rng,
                training=training,
            )
            loss = jnp.mean(jnp.square(prediction - batch["y"]))
            return loss, {"mse": loss}, updated_state

        def configure_optimizer(self, *, total_steps: int) -> Any:
            del total_steps
            import optax

            return optax.sgd(0.1)

    data = ArrayDataSource(
        {
            "x": np.ones((16, 2), dtype=np.float32),
            "y": np.ones((16, 1), dtype=np.float32),
        },
        batch_size=4,
    )
    result = JaxTrainer(
        NNXTask(),
        config=TrainerConfig(max_epochs=2),
    ).fit(data)

    assert result.history[-1].loss < result.history[0].loss
    assert result.state.model_state.mutable is not None


def test_nnx_adapter_supports_dropout_batchnorm_and_eval_state() -> None:
    class TinyStatefulNNX(nnx.Module):
        def __init__(self, rngs: nnx.Rngs) -> None:
            self.norm = nnx.BatchNorm(2, rngs=rngs)
            self.dropout = nnx.Dropout(0.5, rngs=rngs)
            self.linear = nnx.Linear(2, 1, rngs=rngs)

        def __call__(self, inputs: jax.Array) -> jax.Array:
            return self.linear(self.dropout(self.norm(inputs)))

    adapter = NNXModelAdapter(
        lambda rng, batch: TinyStatefulNNX(nnx.Rngs(rng)),
    )
    inputs = jnp.asarray(
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0],
         [5.0, 10.0], [6.0, 12.0], [7.0, 14.0], [8.0, 16.0]],
        dtype=jnp.float32,
    )
    state = adapter.initialize(jax.random.key(17), inputs)

    train_output_1, train_state_1 = adapter.apply(
        state,
        inputs,
        jax.random.key(18),
        training=True,
    )
    train_output_2, train_state_2 = adapter.apply(
        train_state_1,
        inputs,
        jax.random.key(19),
        training=True,
    )
    assert not np.allclose(
        np.asarray(train_output_1),
        np.asarray(train_output_2),
    )
    assert not _tree_allclose(
        train_state_1.mutable,
        train_state_2.mutable,
    )

    eval_output, eval_state = adapter.apply(
        train_state_2,
        inputs,
        jax.random.key(20),
        training=False,
    )
    eval_output_repeat, eval_state_repeat = adapter.apply(
        eval_state,
        inputs,
        jax.random.key(21),
        training=False,
    )
    assert np.allclose(
        np.asarray(eval_output),
        np.asarray(eval_output_repeat),
    )
    assert _tree_allclose(eval_state.mutable, eval_state_repeat.mutable)
