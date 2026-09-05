"""JAX 单设备训练器."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass
from itertools import chain
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .callbacks import Callback, CallbackList
from .data import BatchSpec, validate_batch
from .distributed import JaxStrategy, SingleDeviceStrategy
from .errors import DataValidationError, TrainingError
from .model import ModelAdapter
from .optimizer import build_optimizer
from .task import JaxTask
from .types import (
    JaxTrainerState,
    JaxTrainState,
    MetricSnapshot,
    StepOutput,
    TrainResult,
)


@dataclass(frozen=True)
class TrainerConfig:
    """JAX runtime 配置."""

    max_epochs: int = 1
    gradient_accumulation_steps: int = 1
    drop_incomplete_accumulation: bool = True
    grad_clip_max_norm: float | None = None
    precision: str = "32"
    seed: int = 42
    jit: bool = True
    fail_on_callback_error: bool = False
    log_every_n_steps: int = 1
    nan_patience: int = 0
    validate_every_n_epochs: int = 1
    check_val_every_n_epoch: int | None = None
    max_train_steps: int | None = None
    platform: str = "cpu"

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        prec = self.precision.lower()
        if prec in {"bf16", "bfloat16"}:
            object.__setattr__(self, "precision", "bfloat16")
        elif prec in {"32", "fp32", "float32"}:
            object.__setattr__(self, "precision", "32")
        else:
            raise ValueError("precision must be '32', 'float32', 'bf16' or 'bfloat16'")
        if self.check_val_every_n_epoch is not None:
            if self.check_val_every_n_epoch < 1:
                raise ValueError("check_val_every_n_epoch must be positive")
            object.__setattr__(self, "validate_every_n_epochs", self.check_val_every_n_epoch)
        if self.grad_clip_max_norm is not None and self.grad_clip_max_norm <= 0:
            raise ValueError("grad_clip_max_norm must be positive")
        if self.platform not in {"auto", "cpu", "gpu", "tpu"}:
            raise ValueError(
                "platform must be one of 'auto', 'cpu', 'gpu' or 'tpu'"
            )


def _tree_zeros_like(value: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, value)


def _tree_add(left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _tree_scale(value: Any, scale: jax.Array | float) -> Any:
    return jax.tree_util.tree_map(lambda x: x * scale, value)


def _tree_all_finite(value: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves:
        return jnp.asarray(True)
    flags = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
    return jnp.all(jnp.stack(flags))


def _block_until_ready(value: Any) -> Any:
    def block(leaf: Any) -> Any:
        return leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf

    return jax.tree_util.tree_map(block, value)


def _host_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)))


class JaxTrainer:
    """面向 immutable PyTree state 的单设备训练器."""

    @classmethod
    def from_setup(cls, setup: Any) -> "JaxTrainer":
        """从 `setup_from_yaml()` 的结果创建 trainer."""

        return cls(
            setup.task,
            config=setup.trainer_config,
            optimizer_config=setup.optimizer_config,
            callbacks=setup.callbacks,
        )

    def __init__(
        self,
        task: JaxTask,
        *,
        config: TrainerConfig | None = None,
        optimizer_config: Mapping[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        strategy: JaxStrategy | None = None,
    ) -> None:
        self.task = task
        self.config = config or TrainerConfig()
        self.optimizer_config = optimizer_config
        all_callbacks = list(callbacks or [])
        configure_callbacks = getattr(task, "configure_callbacks", None)
        if callable(configure_callbacks):
            task_callbacks = configure_callbacks()
            if isinstance(task_callbacks, Iterable):
                all_callbacks.extend(task_callbacks)
        self.callbacks = CallbackList(
            all_callbacks,
            fail_on_error=self.config.fail_on_callback_error,
        )
        self.strategy = strategy or SingleDeviceStrategy(
            platform=self.config.platform,
        )
        self._adapter: ModelAdapter | None = None
        self._optimizer: optax.GradientTransformation | None = None
        self._train_step: Any = None
        self._eval_step: Any = None
        self._predict_step_fn: Any = None
        self._batch_spec: BatchSpec | None = None
        self._stop_requested = False
        self._first_compile_time_s: float | None = None
        self._compile_report_emitted = False
        self._nonfinite_count = 0
        self.train_data: Any = None
        self._current_epoch: int = 0
        self._global_step: int = 0
        self._current_step: int = 0

    @property
    def adapter(self) -> ModelAdapter:
        if self._adapter is None:
            raise TrainingError("trainer has not been initialized")
        return self._adapter

    @property
    def model(self) -> ModelAdapter:
        """对齐 PyTorch Trainer.model 别名."""
        return self.adapter

    @property
    def current_epoch(self) -> int:
        """当前训练 Epoch 计数(0-indexed)."""
        return self._current_epoch

    @property
    def global_step(self) -> int:
        """当前累计的优化器步数 optimizer_step."""
        return self._global_step

    @property
    def current_step(self) -> int:
        """当前累计的 micro_step."""
        return self._current_step

    @property
    def optimizer(self) -> optax.GradientTransformation:
        if self._optimizer is None:
            raise TrainingError("trainer optimizer has not been initialized")
        return self._optimizer

    def request_stop(self) -> None:
        """请求在当前 epoch 结束后停止."""

        self._stop_requested = True

    def strategy_report(self) -> Mapping[str, Any]:
        """返回当前设备策略报告."""

        self.strategy.setup()
        return self.strategy.report()

    def _prepare_batch(self, batch: Any) -> Any:
        if self.config.precision != "bfloat16":
            return batch

        def cast_leaf(leaf: Any) -> Any:
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
                return jnp.asarray(leaf, dtype=jnp.bfloat16)
            return leaf

        return jax.tree_util.tree_map(cast_leaf, batch)

    def _initialize(
        self,
        first_batch: Any,
        *,
        total_steps: int,
        state: JaxTrainState | None,
    ) -> JaxTrainState:
        self._adapter = self.task.build_model()
        init_key, train_key = jax.random.split(
            jax.random.PRNGKey(self.config.seed)
        )
        if state is None:
            model_state = self.adapter.initialize(init_key, first_batch)
        else:
            model_state = state.model_state
            train_key = state.rng_key
        if self.optimizer_config is None:
            self._optimizer = self.task.configure_optimizer(total_steps=total_steps)
        else:
            self._optimizer = build_optimizer(
                self.optimizer_config,
                total_steps=total_steps,
            )
        if self.config.grad_clip_max_norm is not None:
            self._optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.grad_clip_max_norm),
                self._optimizer,
            )
        if state is not None:
            return state
        params = self.adapter.get_params(model_state)
        return JaxTrainState(
            model_state=model_state,
            optimizer_state=self.optimizer.init(params),
            rng_key=train_key,
            micro_step=0,
            optimizer_step=0,
            epoch=0,
            accumulation_grads=_tree_zeros_like(params),
            accumulation_count=0,
        )

    def _make_train_step(self, *, reduce_across_devices: bool) -> Any:
        adapter = self.adapter
        task = self.task
        optimizer = self.optimizer
        accumulation_steps = self.config.gradient_accumulation_steps

        def train_step(state: JaxTrainState, batch: Any) -> tuple[JaxTrainState, StepOutput]:
            step_key, next_key = jax.random.split(state.rng_key)
            params = adapter.get_params(state.model_state)

            def loss_fn(current_params: Any) -> tuple[jax.Array, Any]:
                candidate = adapter.with_params(state.model_state, current_params)
                loss, metrics, updated_model = task.loss_and_metrics(
                    adapter,
                    candidate,
                    batch,
                    step_key,
                    training=True,
                )
                return loss, (metrics, updated_model)

            (loss, (metrics, updated_model)), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            if reduce_across_devices:
                grads = self.strategy.reduce_gradients(grads)
                loss = self.strategy.reduce_metrics(loss)
                metrics = self.strategy.reduce_metrics(metrics)
                updated_model = updated_model.replace(
                    mutable=self.strategy.reduce_metrics(updated_model.mutable)
                )
            loss_finite = jnp.isfinite(loss)
            gradients_finite = _tree_all_finite(grads)
            all_finite = jnp.logical_and(loss_finite, gradients_finite)
            gradient_norm = optax.tree.norm(grads)
            next_micro = state.micro_step + 1
            next_count = state.accumulation_count + 1
            boundary = next_count >= accumulation_steps

            def finite_branch() -> tuple[JaxTrainState, StepOutput]:
                accumulated = _tree_add(state.accumulation_grads, grads)

                def update_branch() -> tuple[JaxTrainState, StepOutput]:
                    mean_grads = _tree_scale(
                        accumulated,
                        1.0 / float(accumulation_steps),
                    )
                    updates, optimizer_state = optimizer.update(
                        mean_grads,
                        state.optimizer_state,
                        params,
                    )
                    new_params = optax.apply_updates(params, updates)
                    new_model = adapter.with_params(updated_model, new_params)
                    new_state = state.replace(
                        model_state=new_model,
                        optimizer_state=optimizer_state,
                        rng_key=next_key,
                        micro_step=next_micro,
                        optimizer_step=state.optimizer_step + 1,
                        accumulation_grads=_tree_zeros_like(accumulated),
                        accumulation_count=0,
                    )
                    output = StepOutput(
                        loss=loss,
                        metrics=metrics,
                        did_optimizer_step=jnp.asarray(True),
                        loss_finite=loss_finite,
                        gradients_finite=gradients_finite,
                        gradient_norm=gradient_norm,
                    )
                    return new_state, output

                def accumulate_branch() -> tuple[JaxTrainState, StepOutput]:
                    new_state = state.replace(
                        model_state=updated_model,
                        rng_key=next_key,
                        micro_step=next_micro,
                        accumulation_grads=accumulated,
                        accumulation_count=next_count,
                    )
                    output = StepOutput(
                        loss=loss,
                        metrics=metrics,
                        did_optimizer_step=jnp.asarray(False),
                        loss_finite=loss_finite,
                        gradients_finite=gradients_finite,
                        gradient_norm=gradient_norm,
                    )
                    return new_state, output

                return jax.lax.cond(boundary, update_branch, accumulate_branch)

            def invalid_branch() -> tuple[JaxTrainState, StepOutput]:
                new_state = state.replace(
                    rng_key=next_key,
                    micro_step=next_micro,
                )
                output = StepOutput(
                    loss=loss,
                    metrics=metrics,
                    did_optimizer_step=jnp.asarray(False),
                    loss_finite=loss_finite,
                    gradients_finite=gradients_finite,
                    gradient_norm=gradient_norm,
                )
                return new_state, output

            return jax.lax.cond(all_finite, finite_branch, invalid_branch)

        return jax.jit(train_step) if self.config.jit else train_step

    def _make_eval_step(self, *, reduce_across_devices: bool) -> Any:
        adapter = self.adapter
        task = self.task

        def eval_step(
            model_state: Any,
            batch: Any,
            rng: jax.Array,
        ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
            val_fn = getattr(task, "validation_loss_and_metrics", None)
            if callable(val_fn):
                val_out = cast(tuple[Any, Any], val_fn(
                    adapter,
                    model_state,
                    batch,
                    rng,
                ))
                loss, metrics = val_out[0], val_out[1]
            else:
                loss, metrics, _updated_model = task.loss_and_metrics(
                    adapter,
                    model_state,
                    batch,
                    rng,
                    training=False,
                )
            if reduce_across_devices:
                loss = self.strategy.reduce_metrics(loss)
                metrics = self.strategy.reduce_metrics(metrics)
            return loss, metrics

        return eval_step

    def _make_predict_step(self) -> Any:
        adapter = self.adapter
        task = self.task

        def predict_step(model_state: Any, batch: Any, rng: jax.Array) -> Any:
            predict_fn = getattr(task, "predict_step", None)
            if callable(predict_fn):
                return predict_fn(adapter, model_state, batch, rng=rng)
            preds, _ = adapter.apply(model_state, batch, rng, training=False)
            return preds

        return jax.jit(predict_step) if self.config.jit else predict_step

    def _snapshot(
        self,
        state: JaxTrainState,
        output: StepOutput,
        *,
        step_time_s: float,
    ) -> MetricSnapshot:
        metrics = {
            str(name): _host_float(value)
            for name, value in dict(jax.device_get(output.metrics)).items()
        }
        return MetricSnapshot(
            epoch=int(state.epoch),
            micro_step=int(np.asarray(jax.device_get(state.micro_step))),
            optimizer_step=int(np.asarray(jax.device_get(state.optimizer_step))),
            metrics=metrics,
            loss=_host_float(output.loss),
            did_optimizer_step=bool(np.asarray(jax.device_get(output.did_optimizer_step))),
            loss_finite=bool(np.asarray(jax.device_get(output.loss_finite))),
            gradients_finite=bool(
                np.asarray(jax.device_get(output.gradients_finite))
            ),
            gradient_norm=_host_float(output.gradient_norm),
            step_time_s=step_time_s,
            compile_time_s=self._first_compile_time_s,
        )

    def _flush_accumulation(self, state: JaxTrainState) -> JaxTrainState:
        if state.accumulation_count == 0:
            return state
        if self.config.drop_incomplete_accumulation:
            return state.replace(
                accumulation_grads=_tree_zeros_like(state.accumulation_grads),
                accumulation_count=0,
            )
        adapter = self.adapter
        optimizer = self.optimizer

        def flush_fn(current: JaxTrainState) -> JaxTrainState:
            params = adapter.get_params(current.model_state)
            denominator = jnp.asarray(current.accumulation_count, dtype=jnp.float32)
            grads = _tree_scale(current.accumulation_grads, 1.0 / denominator)
            updates, optimizer_state = optimizer.update(
                grads,
                current.optimizer_state,
                params,
            )
            model_state = adapter.with_params(
                current.model_state,
                optax.apply_updates(params, updates),
            )
            return current.replace(
                model_state=model_state,
                optimizer_state=optimizer_state,
                optimizer_step=current.optimizer_step + 1,
                accumulation_grads=_tree_zeros_like(current.accumulation_grads),
                accumulation_count=0,
            )

        compiled = jax.jit(flush_fn) if self.config.jit else flush_fn
        return _block_until_ready(compiled(state))

    def validate(
        self,
        state: JaxTrainState,
        val_data: Iterable[Any],
        *,
        epoch: int = 0,
    ) -> dict[str, float]:
        """在不修改训练 state 的情况下运行 validation."""

        if self._eval_step is None:
            raise TrainingError("trainer eval step is not initialized")
        if self._batch_spec is None:
            raise TrainingError("trainer batch specification is not initialized")
        values: list[dict[str, float]] = []
        val_key = jax.random.fold_in(state.rng_key, epoch)
        self.callbacks.invoke("on_validation_start", self)
        self.callbacks.invoke("on_validation_epoch_start", self)
        for batch_index, batch in enumerate(val_data):
            batch = self._prepare_batch(batch)
            validate_batch(batch, self._batch_spec, name="validation batch")
            batch = self.strategy.place_batch(batch)
            batch_key = jax.random.fold_in(val_key, batch_index)
            self.callbacks.invoke("on_validation_batch_start", self, batch_index)
            loss, metrics = self._eval_step(state.model_state, batch, batch_key)
            result = {
                str(name): _host_float(value)
                for name, value in dict(jax.device_get(metrics)).items()
            }
            result.setdefault("loss", _host_float(loss))
            self.callbacks.invoke("on_validation_batch_end", self, batch_index, result)
            values.append(result)
        self.callbacks.invoke("on_validation_end", self)
        if not values:
            return {}
        names = sorted({name for item in values for name in item})
        return {
            name: float(np.mean([item[name] for item in values if name in item]))
            for name in names
        }

    def test(
        self,
        test_data: Iterable[Any],
        *,
        state: JaxTrainState | None = None,
    ) -> dict[str, float]:
        """在测试集上评估指标 (对齐 PyTorch Trainer.test)."""
        target_state = state if state is not None else getattr(self, "_last_state", None)
        if target_state is None:
            raise TrainingError("test requires an explicit state or a prior fit() run")
        return self.validate(target_state, test_data, epoch=self._current_epoch)

    def predict(
        self,
        data: Iterable[Any],
        *,
        state: JaxTrainState | None = None,
    ) -> list[Any]:
        """批量推理预测 (对齐 PyTorch Trainer.predict)."""
        target_state = state if state is not None else getattr(self, "_last_state", None)
        if target_state is None:
            raise TrainingError("predict requires an explicit state or a prior fit() run")
        if self._predict_step_fn is None:
            self._predict_step_fn = self._make_predict_step()
        self.callbacks.invoke("on_predict_start", self)
        predictions: list[Any] = []
        pred_key = jax.random.fold_in(target_state.rng_key, 9999)
        for batch_index, batch in enumerate(data):
            batch = self._prepare_batch(batch)
            batch = self.strategy.place_batch(batch)
            batch_key = jax.random.fold_in(pred_key, batch_index)
            output = self._predict_step_fn(target_state.model_state, batch, batch_key)
            _block_until_ready(output)
            predictions.append(jax.device_get(output))
        self.callbacks.invoke("on_predict_end", self)
        return predictions

    def fit(
        self,
        train_data: Iterable[Any],
        *,
        val_data: Iterable[Any] | None = None,
        state: JaxTrainState | None = None,
    ) -> TrainResult:
        """运行单设备训练."""

        self.train_data = train_data
        self.strategy.setup()
        start_epoch = int(state.epoch) if state is not None else 0
        if hasattr(train_data, "set_epoch"):
            train_data.set_epoch(start_epoch)  # type: ignore[attr-defined]
        iterator = iter(train_data)
        try:
            first_batch = next(iterator)
        except StopIteration as exc:
            raise DataValidationError("train_data is empty") from exc
        first_batch = self._prepare_batch(first_batch)
        self._batch_spec = BatchSpec.from_batch(first_batch)
        first_batch = self.strategy.place_batch(first_batch)
        train_batches = len(train_data) if isinstance(train_data, Sized) else None
        total_steps = max(1, (train_batches or 1) * self.config.max_epochs)
        state = cast(
            JaxTrainState,
            self._initialize(first_batch, total_steps=total_steps, state=state),
        )
        state = self.strategy.initialize_state(state)
        local_train_step = self._make_train_step(reduce_across_devices=False)
        distributed_train_step = self._make_train_step(
            reduce_across_devices=True
        )
        output_example = jax.eval_shape(
            local_train_step,
            state,
            first_batch,
        )
        self._train_step = self.strategy.compile_train_step(
            distributed_train_step,
            state,
            first_batch,
            output_example,
            jit=self.config.jit,
        )
        local_eval_step = self._make_eval_step(reduce_across_devices=False)
        distributed_eval_step = self._make_eval_step(reduce_across_devices=True)
        eval_output_example = jax.eval_shape(
            local_eval_step,
            state.model_state,
            first_batch,
            jax.random.PRNGKey(self.config.seed),
        )
        self._eval_step = self.strategy.compile_eval_step(
            distributed_eval_step,
            state.model_state,
            first_batch,
            eval_output_example,
            jit=self.config.jit,
        )
        trainer_state = JaxTrainerState(
            epoch=int(state.epoch),
            micro_step=int(state.micro_step),
            optimizer_step=int(state.optimizer_step),
        )
        history: list[MetricSnapshot] = []
        validation_history: list[dict[str, float]] = []
        self._current_epoch = int(state.epoch)
        self._global_step = int(state.optimizer_step)
        self._current_step = int(state.micro_step)
        self.callbacks.invoke("on_fit_start", self, state)
        self.callbacks.invoke("on_train_start", self)

        for epoch in range(int(state.epoch), self.config.max_epochs):
            self._current_epoch = epoch
            if hasattr(train_data, "set_epoch"):
                train_data.set_epoch(epoch)  # type: ignore[attr-defined]
            state = state.replace(epoch=epoch)
            self.callbacks.invoke("on_train_epoch_start", self, state)
            epoch_data: Iterable[Any]
            if epoch == start_epoch:
                epoch_data = chain((first_batch,), iterator)
            else:
                epoch_data = train_data
            for batch_index, batch in enumerate(epoch_data):
                self.callbacks.invoke("on_train_batch_start", self, batch_index)
                batch = self._prepare_batch(batch)
                validate_batch(batch, self._batch_spec)
                batch = self.strategy.place_batch(batch)
                started = time.perf_counter()
                previous_step = self._first_compile_time_s
                next_state, output = self._train_step(state, batch)
                state = cast(JaxTrainState, next_state)
                _block_until_ready((state, output))
                elapsed = time.perf_counter() - started
                if previous_step is None and self.config.jit:
                    self._first_compile_time_s = elapsed
                    if not self._compile_report_emitted:
                        self.callbacks.invoke(
                            "on_compile_end",
                            self,
                            {
                                "first_compile_time_s": elapsed,
                                "strategy": dict(self.strategy.report()),
                            },
                        )
                        self._compile_report_emitted = True
                snapshot = self._snapshot(state, output, step_time_s=elapsed)
                history.append(snapshot)
                self._global_step = snapshot.optimizer_step
                self._current_step = snapshot.micro_step
                if not snapshot.loss_finite or not snapshot.gradients_finite:
                    self._nonfinite_count += 1
                    if self._nonfinite_count > self.config.nan_patience:
                        raise TrainingError(
                            "non-finite loss or gradients encountered at "
                            f"micro_step={snapshot.micro_step}"
                        )
                else:
                    self._nonfinite_count = 0
                self.callbacks.invoke("on_train_batch_end", self, snapshot)
                trainer_state = JaxTrainerState(
                    epoch=epoch,
                    micro_step=snapshot.micro_step,
                    optimizer_step=snapshot.optimizer_step,
                )
                if (
                    self.config.max_train_steps is not None
                    and snapshot.optimizer_step >= self.config.max_train_steps
                ):
                    self.request_stop()
                    break
            state = self._flush_accumulation(cast(JaxTrainState, state))
            state = state.replace(epoch=epoch + 1)
            self._current_epoch = epoch + 1
            if hasattr(train_data, "set_epoch"):
                train_data.set_epoch(epoch + 1)  # type: ignore[attr-defined]
            trainer_state = JaxTrainerState(
                epoch=epoch + 1,
                micro_step=int(state.micro_step),
                optimizer_step=int(state.optimizer_step),
                stop_requested=self._stop_requested,
            )
            if (
                val_data is not None
                and (epoch + 1) % self.config.validate_every_n_epochs == 0
            ):
                metrics = self.validate(state, val_data, epoch=epoch)
                validation_history.append(metrics)
                self.callbacks.invoke("on_validation_epoch_end", self, metrics)
            self.callbacks.invoke("on_train_epoch_end", self, state)
            if self._stop_requested:
                break

        self.callbacks.invoke("on_train_end", self)
        self._last_state = state
        result = TrainResult(
            state=state,
            trainer_state=trainer_state,
            history=tuple(history),
            validation_history=tuple(validation_history),
            first_compile_time_s=self._first_compile_time_s,
        )
        self.callbacks.invoke("on_fit_end", self, result)
        return result

    def fit_from_setup(self, setup: Any) -> TrainResult:
        """消费 `JaxTrainSetup`, 保持 YAML 入口与 Python 入口一致."""

        if setup.task is not self.task:
            raise TrainingError("setup task does not match this trainer")
        self.optimizer_config = setup.optimizer_config
        self.callbacks = CallbackList(
            setup.callbacks,
            fail_on_error=self.config.fail_on_callback_error,
        )
        return self.fit(setup.train_data, val_data=setup.val_data)
