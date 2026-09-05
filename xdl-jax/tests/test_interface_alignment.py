"""对齐 PyTorch 接口特性的全面测试."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest

from xdl_jax import (
    BaseJaxTask,
    Callback,
    FunctionalModelAdapter,
    JaxTrainer,
    TrainerConfig,
)
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData


class AlignmentRecordingCallback(Callback):
    """记录完整生命周期钩子的测试回调."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self, trainer: Any) -> None:
        self.events.append("on_train_start")

    def on_train_end(self, trainer: Any) -> None:
        self.events.append("on_train_end")

    def on_train_batch_start(self, trainer: Any, batch_idx: int) -> None:
        self.events.append(f"on_train_batch_start_{batch_idx}")

    def on_validation_start(self, trainer: Any) -> None:
        self.events.append("on_validation_start")

    def on_validation_end(self, trainer: Any) -> None:
        self.events.append("on_validation_end")

    def on_validation_batch_start(self, trainer: Any, batch_idx: int) -> None:
        self.events.append(f"on_validation_batch_start_{batch_idx}")

    def on_predict_start(self, trainer: Any) -> None:
        self.events.append("on_predict_start")

    def on_predict_end(self, trainer: Any) -> None:
        self.events.append("on_predict_end")


class CustomBaseTask(BaseJaxTask):
    """基于 BaseJaxTask 实现的任务, 重写自定义验证步和预测步."""

    def __init__(self, input_dim: int = 2) -> None:
        self.input_dim = input_dim
        self.callback = AlignmentRecordingCallback()

    def build_model(self) -> FunctionalModelAdapter:
        def init_fn(rng: jax.Array, sample_batch: Any) -> dict[str, jax.Array]:
            del sample_batch
            return {"w": jax.random.normal(rng, (self.input_dim, 1)), "b": jnp.zeros((1,))}

        def apply_fn(
            params: Any,
            mutable: Any,
            batch: Any,
            rng: jax.Array,
            training: bool,
        ) -> Any:
            del mutable, rng, training
            return jnp.matmul(batch["x"], params["w"]) + params["b"]

        return FunctionalModelAdapter(init_fn=init_fn, apply_fn=apply_fn)

    def loss_and_metrics(self, model, model_state, batch, rng, *, training: bool):
        preds, _ = model.apply(model_state, batch, rng, training=training)
        loss = jnp.mean((preds - batch["y"]) ** 2)
        return loss, {"loss": loss}, model_state

    def validation_loss_and_metrics(self, model, model_state, batch, rng):
        preds, _ = model.apply(model_state, batch, rng, training=False)
        loss = jnp.mean((preds - batch["y"]) ** 2)
        return loss, {"loss": loss, "custom_val_metric": 1.0}

    def configure_optimizer(self, *, total_steps: int):
        del total_steps
        return optax.adam(learning_rate=0.01)

    def configure_callbacks(self) -> list[Callback]:
        return [self.callback]


def test_trainer_properties_and_lifecycle():
    data = SyntheticRegressionData(n_samples=16, input_dim=2, batch_size=4, seed=1)
    val_data = SyntheticRegressionData(n_samples=8, input_dim=2, batch_size=4, seed=2)

    cb = AlignmentRecordingCallback()
    trainer = JaxTrainer(
        LinearRegressionTask(input_dim=2),
        config=TrainerConfig(max_epochs=2, platform="cpu", validate_every_n_epochs=1),
        callbacks=[cb],
    )

    result = trainer.fit(data, val_data=val_data)

    # 验证属性对齐
    assert trainer.current_epoch == 2
    assert trainer.global_step == result.state.optimizer_step
    assert trainer.current_step == result.state.micro_step
    assert trainer.model is trainer.adapter

    # 验证生命周期钩子
    assert "on_train_start" in cb.events
    assert "on_train_end" in cb.events
    assert "on_train_batch_start_0" in cb.events
    assert "on_validation_start" in cb.events
    assert "on_validation_end" in cb.events
    assert "on_validation_batch_start_0" in cb.events


def test_base_jax_task_custom_validation_and_callbacks():
    task = CustomBaseTask(input_dim=2)
    data = SyntheticRegressionData(n_samples=16, input_dim=2, batch_size=4, seed=1)
    val_data = SyntheticRegressionData(n_samples=8, input_dim=2, batch_size=4, seed=2)

    trainer = JaxTrainer(
        task,
        config=TrainerConfig(max_epochs=1, platform="cpu", validate_every_n_epochs=1),
    )

    # 自动收集 task.configure_callbacks()
    assert any(isinstance(c, AlignmentRecordingCallback) for c in trainer.callbacks.callbacks)

    result = trainer.fit(data, val_data=val_data)
    assert len(result.validation_history) == 1
    # 验证自定义验证步指标生效
    assert "custom_val_metric" in result.validation_history[0]


def test_trainer_test_and_predict():
    task = LinearRegressionTask(input_dim=2)
    data = SyntheticRegressionData(n_samples=16, input_dim=2, batch_size=4, seed=1)
    test_data = SyntheticRegressionData(n_samples=8, input_dim=2, batch_size=4, seed=3)

    cb = AlignmentRecordingCallback()
    trainer = JaxTrainer(
        task,
        config=TrainerConfig(max_epochs=1, platform="cpu"),
        callbacks=[cb],
    )

    trainer.fit(data)

    # 测试 test 方法
    test_metrics = trainer.test(test_data)
    assert "loss" in test_metrics

    # 测试 predict 方法
    preds = trainer.predict(test_data)
    assert len(preds) == 2  # 8 samples / batch_size 4
    assert preds[0].shape == (4, 1)

    assert "on_predict_start" in cb.events
    assert "on_predict_end" in cb.events


def test_trainer_config_precision_and_val_check_aliases():
    cfg = TrainerConfig(precision="bf16", check_val_every_n_epoch=2)
    assert cfg.precision == "bfloat16"
    assert cfg.validate_every_n_epochs == 2

    cfg2 = TrainerConfig(precision="fp32")
    assert cfg2.precision == "32"

    with pytest.raises(ValueError):
        TrainerConfig(precision="invalid")
