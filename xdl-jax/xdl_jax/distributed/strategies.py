"""单设备和单主机 data parallel 的策略边界."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import numpy as np

from ..errors import DataValidationError, TrainingError
from ..types import JaxTrainState, ModelState

JaxDevice = Any


def _validate_platform(platform: str) -> str:
    if platform not in {"auto", "cpu", "gpu", "tpu"}:
        raise ValueError(
            "platform must be one of 'auto', 'cpu', 'gpu' or 'tpu'"
        )
    return platform


def _available_devices(platform: str) -> tuple[JaxDevice, ...]:
    platform = _validate_platform(platform)
    try:
        return tuple(jax.devices()) if platform == "auto" else tuple(
            jax.devices(platform)
        )
    except RuntimeError as exc:
        raise TrainingError(
            f"requested JAX platform '{platform}' is not available; "
            "install the matching JAX accelerator plugin and verify the driver"
        ) from exc


@dataclass(frozen=True)
class MeshConfig:
    """单主机 mesh 配置."""

    axis_name: str = "data"
    require_multiple_devices: bool = True
    device_ids: tuple[int, ...] | None = None
    platform: str = "auto"

    def __post_init__(self) -> None:
        _validate_platform(self.platform)


@dataclass(frozen=True)
class DeviceReport:
    """设备拓扑和策略能力报告."""

    platform: str
    process_index: int
    process_count: int
    visible_devices: tuple[str, ...]
    selected_devices: tuple[str, ...]
    strategy: str
    axis_names: tuple[str, ...]
    supports_data_parallel: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "process_index": self.process_index,
            "process_count": self.process_count,
            "visible_devices": list(self.visible_devices),
            "selected_devices": list(self.selected_devices),
            "strategy": self.strategy,
            "axis_names": list(self.axis_names),
            "supports_data_parallel": self.supports_data_parallel,
        }


class JaxStrategy(Protocol):
    """Trainer 使用的 host-side device strategy."""

    def setup(self) -> None:
        """检查设备并创建必要的 mesh."""
        ...

    def place_batch(self, batch: Any) -> Any:
        """将 host batch 放到策略要求的 sharding."""
        ...

    def initialize_state(self, state: JaxTrainState) -> JaxTrainState:
        """将初始 state 放到策略要求的设备."""
        ...

    def report(self) -> Mapping[str, Any]:
        """返回可序列化的设备报告."""
        ...

    def reduce_gradients(self, gradients: Any) -> Any:
        """在 compiled step 中归约梯度."""
        ...

    def reduce_metrics(self, metrics: Any) -> Any:
        """在 compiled step 中归约 loss 或 metrics."""
        ...

    def compile_train_step(
        self,
        step_fn: Callable[..., Any],
        state_example: JaxTrainState,
        batch_example: Any,
        output_example: Any,
        *,
        jit: bool,
    ) -> Callable[..., Any]:
        """将局部 train step 编译成策略对应的 executable."""
        ...

    def compile_eval_step(
        self,
        step_fn: Callable[..., Any],
        state_example: ModelState,
        batch_example: Any,
        output_example: Any,
        *,
        jit: bool,
    ) -> Callable[..., Any]:
        """将局部 eval step 编译成策略对应的 executable."""
        ...


def _device_name(device: JaxDevice) -> str:
    return f"{device.platform}:{device.id}"


def _device_put_tree(batch: Any, put_leaf: Any) -> Any:
    return jax.tree_util.tree_map(put_leaf, batch)


class SingleDeviceStrategy:
    """CPU/GPU/TPU 单设备策略."""

    def __init__(
        self,
        device: JaxDevice | None = None,
        *,
        platform: str = "auto",
        device_id: int = 0,
    ) -> None:
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        self.device = device
        self.platform = _validate_platform(platform)
        self.device_id = device_id
        self._report: DeviceReport | None = None

    def setup(self) -> None:
        devices = (
            (self.device,)
            if self.device is not None
            else _available_devices(self.platform)
        )
        if not devices:
            raise TrainingError(
                f"requested JAX platform '{self.platform}' has no devices"
            )
        if self.device_id >= len(devices):
            raise TrainingError(
                f"device_id={self.device_id} is unavailable for platform "
                f"'{self.platform}' with {len(devices)} device(s)"
            )
        selected = devices[self.device_id]
        self.device = selected
        visible_devices = _available_devices(self.platform)
        self._report = DeviceReport(
            platform=selected.platform,
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            visible_devices=tuple(
                _device_name(item) for item in visible_devices
            ),
            selected_devices=(_device_name(selected),),
            strategy="single_device",
            axis_names=(),
            supports_data_parallel=False,
        )

    def place_batch(self, batch: Any) -> Any:
        if self.device is None:
            self.setup()
        return _device_put_tree(batch, lambda leaf: jax.device_put(leaf, self.device))

    def initialize_state(self, state: JaxTrainState) -> JaxTrainState:
        if self.device is None:
            self.setup()
        return jax.device_put(state, self.device)

    def report(self) -> Mapping[str, Any]:
        if self._report is None:
            self.setup()
        return self._report.to_dict()  # type: ignore[union-attr]

    def reduce_gradients(self, gradients: Any) -> Any:
        return gradients

    def reduce_metrics(self, metrics: Any) -> Any:
        return metrics

    def compile_train_step(
        self,
        step_fn: Callable[..., Any],
        state_example: JaxTrainState,
        batch_example: Any,
        output_example: Any,
        *,
        jit: bool,
    ) -> Callable[..., Any]:
        del state_example, batch_example, output_example
        return jax.jit(step_fn) if jit else step_fn

    def compile_eval_step(
        self,
        step_fn: Callable[..., Any],
        state_example: ModelState,
        batch_example: Any,
        output_example: Any,
        *,
        jit: bool,
    ) -> Callable[..., Any]:
        del state_example, batch_example, output_example
        return jax.jit(step_fn) if jit else step_fn


class DataParallelStrategy:
    """单主机一维 data mesh 的 placement 基础实现.

    该类负责 mesh,batch sharding,state placement 和 compiled step 中的
    梯度/指标 collective. `supports_data_parallel` 只描述当前拓扑是否具备
    至少两个设备, 不代表多主机或跨拓扑 checkpoint 已得到支持.
    """

    def __init__(self, config: MeshConfig | None = None) -> None:
        self.config = config or MeshConfig()
        self.devices: tuple[JaxDevice, ...] = ()
        self.mesh: jax.sharding.Mesh | None = None
        self._report: DeviceReport | None = None

    def setup(self) -> None:
        visible = _available_devices(self.config.platform)
        if self.config.device_ids is None:
            selected = visible
        else:
            selected = tuple(
                visible[index]
                for index in self.config.device_ids
                if 0 <= index < len(visible)
            )
            if len(selected) != len(self.config.device_ids):
                raise TrainingError("device_ids contains an unavailable device")
        if not selected:
            raise TrainingError("no JAX device is available")
        if self.config.require_multiple_devices and len(selected) < 2:
            raise TrainingError(
                "data parallel requires at least two devices; "
                f"found {len(selected)}"
            )
        self.devices = selected
        self.mesh = jax.sharding.Mesh(
            np.asarray(selected, dtype=object),
            (self.config.axis_name,),
        )
        self._report = DeviceReport(
            platform=selected[0].platform,
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            visible_devices=tuple(_device_name(item) for item in visible),
            selected_devices=tuple(_device_name(item) for item in selected),
            strategy="data_parallel",
            axis_names=(self.config.axis_name,),
            supports_data_parallel=len(selected) >= 2,
        )

    def _require_setup(self) -> None:
        if self.mesh is None:
            self.setup()

    def validate_batch(self, batch: Any) -> None:
        self._require_setup()
        leaves = jax.tree_util.tree_leaves(batch)
        for index, leaf in enumerate(leaves):
            if not hasattr(leaf, "shape"):
                raise DataValidationError(f"batch leaf {index} is not an array")
            if leaf.ndim == 0:
                continue
            if leaf.shape[0] % len(self.devices) != 0:
                raise DataValidationError(
                    f"batch leaf {index} first dimension {leaf.shape[0]} "
                    f"is not divisible by {len(self.devices)} devices"
                )

    def place_batch(self, batch: Any) -> Any:
        self.validate_batch(batch)
        mesh = self.mesh
        if mesh is None:
            raise TrainingError("data parallel mesh has not been initialized")

        def place(leaf: Any) -> Any:
            if leaf.ndim == 0:
                spec = jax.sharding.PartitionSpec()
            else:
                spec = jax.sharding.PartitionSpec(
                    self.config.axis_name,
                    *([None] * (leaf.ndim - 1)),
                )
            sharding = jax.sharding.NamedSharding(mesh, spec)
            return jax.device_put(leaf, sharding)

        return _device_put_tree(batch, place)

    def initialize_state(self, state: JaxTrainState) -> JaxTrainState:
        self._require_setup()
        mesh = self.mesh
        if mesh is None:
            raise TrainingError("data parallel mesh has not been initialized")

        def replicate(leaf: Any) -> Any:
            if not hasattr(leaf, "ndim"):
                return leaf
            spec = jax.sharding.PartitionSpec(*([None] * leaf.ndim))
            return jax.device_put(
                leaf,
                jax.sharding.NamedSharding(mesh, spec),
            )

        return _device_put_tree(state, replicate)

    def report(self) -> Mapping[str, Any]:
        self._require_setup()
        return self._report.to_dict()  # type: ignore[union-attr]

    def _pmean_tree(self, value: Any) -> Any:
        if value is None:
            return None
        return jax.tree_util.tree_map(
            lambda leaf: jax.lax.pmean(leaf, axis_name=self.config.axis_name),
            value,
        )

    def reduce_gradients(self, gradients: Any) -> Any:
        return self._pmean_tree(gradients)

    def reduce_metrics(self, metrics: Any) -> Any:
        return self._pmean_tree(metrics)

    @staticmethod
    def _replicated_specs(value: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda _leaf: jax.sharding.PartitionSpec(),
            value,
        )

    def _batch_specs(self, batch: Any) -> Any:
        def leaf_spec(leaf: Any) -> jax.sharding.PartitionSpec:
            if not hasattr(leaf, "ndim") or leaf.ndim == 0:
                return jax.sharding.PartitionSpec()
            return jax.sharding.PartitionSpec(
                self.config.axis_name,
                *([None] * (leaf.ndim - 1)),
            )

        return jax.tree_util.tree_map(leaf_spec, batch)

    def compile_train_step(
        self,
        step_fn: Callable[..., Any],
        state_example: JaxTrainState,
        batch_example: Any,
        output_example: Any,
        *,
        jit: bool,
    ) -> Callable[..., Any]:
        self._require_setup()
        mesh = self.mesh
        if mesh is None:
            raise TrainingError("data parallel mesh has not been initialized")
        in_specs = (
            self._replicated_specs(state_example),
            self._batch_specs(batch_example),
        )
        out_specs = self._replicated_specs(output_example)
        try:
            mapped = jax.shard_map(
                step_fn,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=True,
            )
        except TypeError:
            from jax.experimental.shard_map import shard_map

            mapped = shard_map(
                step_fn,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=True,
            )
        return jax.jit(mapped) if jit else mapped

    def compile_eval_step(
        self,
        step_fn: Callable[..., Any],
        state_example: ModelState,
        batch_example: Any,
        output_example: Any,
        *,
        jit: bool,
    ) -> Callable[..., Any]:
        self._require_setup()
        mesh = self.mesh
        if mesh is None:
            raise TrainingError("data parallel mesh has not been initialized")
        in_specs = (
            self._replicated_specs(state_example),
            self._batch_specs(batch_example),
            self._replicated_specs(jax.random.PRNGKey(0)),
        )
        out_specs = self._replicated_specs(output_example)
        try:
            mapped = jax.shard_map(
                step_fn,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=True,
            )
        except TypeError:
            from jax.experimental.shard_map import shard_map

            mapped = shard_map(
                step_fn,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=True,
            )
        return jax.jit(mapped) if jit else mapped
