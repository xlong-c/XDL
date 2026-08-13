"""CPU synthetic MLP/linear regression 示例."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..data import ArrayDataSource
from ..model import FunctionalModelAdapter, ModelAdapter
from ..optimizer import build_optimizer
from ..registry import register_dataset, register_task
from ..task import JaxTask
from ..types import ModelState


@register_dataset("SyntheticRegressionData")
class SyntheticRegressionData(ArrayDataSource):
    """生成 y = x @ weight + bias 的可复现数据."""

    def __init__(
        self,
        *,
        n_samples: int = 128,
        input_dim: int = 4,
        output_dim: int = 1,
        batch_size: int = 16,
        seed: int = 0,
        noise: float = 0.0,
        shuffle: bool = True,
        drop_remainder: bool = True,
    ) -> None:
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(n_samples, input_dim)).astype(np.float32)
        weight = rng.normal(size=(input_dim, output_dim)).astype(np.float32)
        bias = rng.normal(size=(output_dim,)).astype(np.float32)
        targets = features @ weight + bias
        if noise:
            targets += rng.normal(
                scale=noise,
                size=targets.shape,
            ).astype(np.float32)
        super().__init__(
            {"x": features, "y": targets},
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed + 1,
            drop_remainder=drop_remainder,
        )


@register_task("LinearRegressionTask")
class LinearRegressionTask(JaxTask):
    """用于验证 runtime 的最小函数式任务."""

    def __init__(
        self,
        *,
        input_dim: int = 4,
        output_dim: int = 1,
        learning_rate: float = 0.05,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.learning_rate = float(learning_rate)

    def build_model(self) -> ModelAdapter:
        input_dim = self.input_dim
        output_dim = self.output_dim

        def init_fn(rng: jax.Array, sample_batch: Any) -> ModelState:
            del sample_batch
            key_w, _key_b = jax.random.split(rng)
            return ModelState(
                params={
                    "weight": jax.random.normal(
                        key_w,
                        (input_dim, output_dim),
                    )
                    * 0.01,
                    "bias": jnp.zeros((output_dim,), dtype=jnp.float32),
                }
            )

        def apply_fn(
            params: Any,
            mutable: Any,
            batch: Any,
            rng: jax.Array,
            training: bool,
        ) -> Any:
            del mutable, rng, training
            return batch["x"] @ params["weight"] + params["bias"]

        return FunctionalModelAdapter(init_fn, apply_fn)

    def loss_and_metrics(
        self,
        model: ModelAdapter,
        model_state: ModelState,
        batch: Any,
        rng: jax.Array,
        *,
        training: bool,
    ) -> tuple[jax.Array, dict[str, jax.Array], ModelState]:
        predictions, new_state = model.apply(
            model_state,
            batch,
            rng,
            training=training,
        )
        loss = jnp.mean(jnp.square(predictions - batch["y"]))
        return loss, {"mse": loss}, new_state

    def configure_optimizer(
        self,
        *,
        total_steps: int,
    ) -> Any:
        return build_optimizer(
            {
                "target": "optax:adamw",
                "params": {
                    "learning_rate": self.learning_rate,
                    "weight_decay": 0.0,
                },
            },
            total_steps=total_steps,
        )
