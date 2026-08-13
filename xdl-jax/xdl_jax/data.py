"""JAX 输入数据协议与 NumPy batch source."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from .errors import DataValidationError


class JaxDataSource(Iterable[Any]):
    """可被 Trainer 重复迭代的 batch source."""

    def set_epoch(self, epoch: int) -> None:
        """设置 deterministic shuffle 使用的 epoch."""

    def state_dict(self) -> dict[str, Any]:
        """返回可序列化的数据源状态."""

        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """恢复数据源状态."""


@dataclass(frozen=True)
class LeafSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class BatchSpec:
    """固定 batch PyTree 的结构, shape 和 dtype."""

    structure: str
    leaves: tuple[LeafSpec, ...]

    @classmethod
    def from_batch(cls, batch: Any) -> "BatchSpec":
        leaves, treedef = jax.tree_util.tree_flatten(batch)
        if not leaves:
            raise DataValidationError("batch must contain at least one leaf")
        specs: list[LeafSpec] = []
        for index, leaf in enumerate(leaves):
            if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
                raise DataValidationError(
                    f"batch leaf {index} must be an array with shape and dtype"
                )
            specs.append(
                LeafSpec(
                    shape=tuple(int(value) for value in leaf.shape),
                    dtype=str(np.dtype(leaf.dtype)),
                )
            )
        return cls(structure=str(treedef), leaves=tuple(specs))


def validate_batch(batch: Any, expected: BatchSpec, *, name: str = "batch") -> None:
    """校验 batch 的 PyTree, dtype 和固定 shape."""

    leaves, treedef = jax.tree_util.tree_flatten(batch)
    if str(treedef) != expected.structure:
        raise DataValidationError(
            f"{name} structure mismatch: expected {expected.structure}, got {treedef}"
        )
    if len(leaves) != len(expected.leaves):
        raise DataValidationError(
            f"{name} leaf count mismatch: expected {len(expected.leaves)}, got {len(leaves)}"
        )
    for index, (leaf, spec) in enumerate(zip(leaves, expected.leaves, strict=True)):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            raise DataValidationError(f"{name} leaf {index} is not an array")
        actual_shape = tuple(int(value) for value in leaf.shape)
        actual_dtype = str(np.dtype(leaf.dtype))
        if actual_shape != spec.shape:
            raise DataValidationError(
                f"{name} leaf {index} shape mismatch: expected {spec.shape}, "
                f"got {actual_shape}"
            )
        if actual_dtype != spec.dtype:
            raise DataValidationError(
                f"{name} leaf {index} dtype mismatch: expected {spec.dtype}, "
                f"got {actual_dtype}"
            )


class ArrayDataSource(JaxDataSource):
    """从同一长度的 NumPy/JAX 数组生成固定 shape batch."""

    def __init__(
        self,
        data: Any,
        *,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_remainder: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise DataValidationError("batch_size must be positive")
        self.data = jax.tree_util.tree_map(np.asarray, data)
        leaves = jax.tree_util.tree_leaves(self.data)
        if not leaves:
            raise DataValidationError("data must contain at least one array")
        lengths = {int(leaf.shape[0]) for leaf in leaves if leaf.ndim > 0}
        if len(lengths) != 1 or any(leaf.ndim == 0 for leaf in leaves):
            raise DataValidationError(
                "all data leaves must be non-scalar arrays with the same first dimension"
            )
        self._size = lengths.pop()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_remainder = bool(drop_remainder)
        self._epoch = 0

    def __len__(self) -> int:
        if self.drop_remainder:
            return self._size // self.batch_size
        return (self._size + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise DataValidationError("epoch must be non-negative")
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[Any]:
        indices = np.arange(self._size)
        if self.shuffle:
            np.random.default_rng(self.seed + self._epoch).shuffle(indices)
        limit = len(self) * self.batch_size if self.drop_remainder else self._size
        for start in range(0, limit, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield jax.tree_util.tree_map(
                lambda values: values[batch_indices],
                self.data,
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "size": self._size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_remainder": self.drop_remainder,
            "exact_iterator_resume": False,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "size": self._size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_remainder": self.drop_remainder,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise DataValidationError(
                    f"data source state mismatch for '{key}': "
                    f"expected {value}, got {state.get(key)}"
                )
        self.set_epoch(int(state.get("epoch", 0)))
