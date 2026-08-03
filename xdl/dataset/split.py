"""Dataset split helpers."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Subset
from torch.utils.data.dataset import random_split as _torch_random_split


def split_dataset(
    dataset: Any,
    lengths: Sequence[int],
    *,
    seed: int = 42,
) -> list[Subset[Any]]:
    """Deterministic random split of *dataset* into subsets.

    Thin wrapper around :func:`torch.utils.data.random_split` that accepts a
    *seed* for reproducibility.
    """

    if sum(lengths) != len(dataset):
        raise ValueError(
            f"Sum of lengths ({sum(lengths)}) does not match dataset size ({len(dataset)})"
        )
    generator = torch.Generator().manual_seed(seed)
    return _torch_random_split(dataset, list(lengths), generator=generator)


def train_val_split(
    dataset: Any,
    val_ratio: float = 0.1,
    *,
    seed: int = 42,
) -> tuple[Any, Any]:
    """Split *dataset* into train and validation subsets.

    Parameters
    ----------
    val_ratio:
        Fraction of samples reserved for validation (clamped to ``[0, 1]``).
    seed:
        Random seed for reproducibility.
    """

    val_ratio = max(0.0, min(1.0, float(val_ratio)))
    if val_ratio == 0.0:
        val_len = 0
    elif val_ratio == 1.0:
        val_len = len(dataset)
    else:
        val_len = max(1, int(len(dataset) * val_ratio))
    train_len = len(dataset) - val_len
    if val_len == 0:
        return dataset, Subset(dataset, [])
    if train_len == 0:
        return Subset(dataset, []), dataset
    train, val = split_dataset(dataset, [train_len, val_len], seed=seed)
    return train, val


__all__ = ["split_dataset", "train_val_split"]
