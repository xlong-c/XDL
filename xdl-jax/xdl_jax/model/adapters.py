"""Functional 和 Flax NNX 模型适配器."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import jax

from ..errors import TrainingError
from ..types import ModelState, PyTree

try:
    from flax import nnx
except ImportError:  # pragma: no cover - 由依赖安装环境决定
    nnx = None  # type: ignore[assignment]


class ModelAdapter(Protocol):
    """训练 runtime 使用的最小模型契约."""

    def initialize(self, rng: jax.Array, sample_batch: Any) -> ModelState:
        """根据随机 key 和样例 batch 初始化模型状态."""
        ...

    def get_params(self, model_state: ModelState) -> PyTree:
        """提取唯一可训练参数 PyTree."""
        ...

    def with_params(self, model_state: ModelState, params: PyTree) -> ModelState:
        """以新参数创建模型状态."""
        ...

    def apply(
        self,
        model_state: ModelState,
        batch: Any,
        rng: jax.Array,
        *,
        training: bool,
    ) -> tuple[Any, ModelState]:
        """执行前向并返回输出及更新后的模型状态."""
        ...

    def state_spec(self, model_state: ModelState) -> Mapping[str, Any]:
        """返回不包含实际数组值的状态摘要."""
        ...


InitFn = Callable[[jax.Array, Any], Any]
FunctionalApplyFn = Callable[[PyTree, PyTree, Any, jax.Array, bool], Any]
NNXFactory = Callable[[jax.Array, Any], Any]
NNXApplyFn = Callable[[Any, Any, jax.Array, bool], Any]


def _normalize_init_value(value: Any) -> ModelState:
    if isinstance(value, ModelState):
        return value
    if isinstance(value, Mapping) and "params" in value:
        mutable = value.get("mutable")
        return ModelState(params=value["params"], mutable=mutable)
    if isinstance(value, tuple) and len(value) == 2:
        return ModelState(params=value[0], mutable=value[1])
    return ModelState(params=value)


def _normalize_apply_value(value: Any, mutable: PyTree) -> tuple[Any, PyTree]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return value, mutable


def _tree_spec(value: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda leaf: {
            "shape": tuple(getattr(leaf, "shape", ())),
            "dtype": str(getattr(leaf, "dtype", type(leaf))),
        },
        value,
    )


class FunctionalModelAdapter:
    """不依赖具体神经网络库的函数式模型 adapter.

    `init_fn` 可以返回 `ModelState`, `(params, mutable)`, 包含 `params` 的
    mapping 或直接返回参数 PyTree. `apply_fn` 返回 `(outputs, mutable)` 时,
    第二项会写回模型状态; 只返回 outputs 时, mutable 保持不变.
    """

    def __init__(
        self,
        init_fn: InitFn,
        apply_fn: FunctionalApplyFn,
    ) -> None:
        self._init_fn = init_fn
        self._apply_fn = apply_fn

    def initialize(self, rng: jax.Array, sample_batch: Any) -> ModelState:
        return _normalize_init_value(self._init_fn(rng, sample_batch))

    def get_params(self, model_state: ModelState) -> PyTree:
        return model_state.params

    def with_params(self, model_state: ModelState, params: PyTree) -> ModelState:
        return ModelState(params=params, mutable=model_state.mutable)

    def apply(
        self,
        model_state: ModelState,
        batch: Any,
        rng: jax.Array,
        *,
        training: bool,
    ) -> tuple[Any, ModelState]:
        value = self._apply_fn(
            model_state.params,
            model_state.mutable,
            batch,
            rng,
            training,
        )
        outputs, mutable = _normalize_apply_value(value, model_state.mutable)
        return outputs, ModelState(params=model_state.params, mutable=mutable)

    def state_spec(self, model_state: ModelState) -> Mapping[str, Any]:
        return {
            "params": _tree_spec(model_state.params),
            "mutable": _tree_spec(model_state.mutable),
        }


class NNXModelAdapter:
    """Flax NNX module adapter.

    NNX 的 `GraphDef` 是静态结构, 参数和其余变量作为动态 PyTree. NNX
    版本相关的 split/merge 细节集中在本类, 不泄漏到 Trainer.
    """

    def __init__(
        self,
        factory: NNXFactory,
        apply_fn: NNXApplyFn | None = None,
    ) -> None:
        if nnx is None:
            raise TrainingError(
                "NNXModelAdapter requires flax; install xdl-jax dependencies"
            )
        self._factory = factory
        self._apply_fn = apply_fn

    @staticmethod
    def _split(module: Any) -> tuple[Any, Any, Any]:
        if nnx is None:  # pragma: no cover
            raise TrainingError("Flax NNX is not available")
        split_result = nnx.split(module, nnx.Param, ...)
        if len(split_result) != 3:
            raise TrainingError(
                "NNX split contract changed: expected GraphDef, params and mutable"
            )
        graphdef, params, mutable = split_result
        return graphdef, params, mutable

    def initialize(self, rng: jax.Array, sample_batch: Any) -> ModelState:
        module = self._factory(rng, sample_batch)
        graphdef, params, mutable = self._split(module)
        return ModelState(params=(graphdef, params), mutable=mutable)

    @staticmethod
    def _unpack_params(params: PyTree) -> tuple[Any, Any]:
        if not isinstance(params, tuple) or len(params) != 2:
            raise TrainingError("NNX params must be the internal (GraphDef, State) pair")
        return params[0], params[1]

    def get_params(self, model_state: ModelState) -> PyTree:
        _graphdef, params = self._unpack_params(model_state.params)
        return params

    def with_params(self, model_state: ModelState, params: PyTree) -> ModelState:
        graphdef, _old_params = self._unpack_params(model_state.params)
        return ModelState(params=(graphdef, params), mutable=model_state.mutable)

    def apply(
        self,
        model_state: ModelState,
        batch: Any,
        rng: jax.Array,
        *,
        training: bool,
    ) -> tuple[Any, ModelState]:
        if nnx is None:  # pragma: no cover
            raise TrainingError("Flax NNX is not available")
        graphdef, params = self._unpack_params(model_state.params)
        # NNX variables are mutable objects.  `copy=True` is required here so
        # a forward pass cannot mutate the variable wrappers retained by the
        # previous immutable ModelState.
        module = nnx.merge(
            graphdef,
            params,
            model_state.mutable,
            copy=True,
        )
        if hasattr(module, "train") and hasattr(module, "eval"):
            if training:
                module.train()
            else:
                module.eval()
        if self._apply_fn is None:
            outputs = module(batch)
        else:
            outputs = self._apply_fn(module, batch, rng, training)
        new_graphdef, new_params, new_mutable = self._split(module)
        # `train()` and `eval()` intentionally update static NNX attributes
        # such as `deterministic` and `use_running_average`.  That produces a
        # different GraphDef without changing the logical module topology.
        # The returned GraphDef is therefore part of the next immutable state.
        return outputs, ModelState(
            params=(new_graphdef, new_params),
            mutable=new_mutable,
        )

    def state_spec(self, model_state: ModelState) -> Mapping[str, Any]:
        graphdef, params = self._unpack_params(model_state.params)
        return {
            "adapter": "nnx",
            "graphdef": type(graphdef).__name__,
            "params": _tree_spec(params),
            "mutable": _tree_spec(model_state.mutable),
        }
