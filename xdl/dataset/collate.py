"""内置 collate 函数."""

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, cast

import torch
from torch.utils.data.dataloader import default_collate

from xdl.utils.registry import COLLATE_REGISTRY


@COLLATE_REGISTRY.register("PadCollate")
class PadCollate:
    """不等长序列 collate - pad 到批次内最大长度.

    Args:
        pad_value: 填充值, 默认 0.
        batch_first: 输出 batch 维度是否在前, 默认 True.
    """

    def __init__(self, pad_value: float = 0.0, batch_first: bool = True) -> None:
        self.pad_value = pad_value
        self.batch_first = batch_first

    def __call__(self, batch: Sequence[Any]) -> Any:
        if isinstance(batch[0], torch.Tensor):
            tensors = cast(list[torch.Tensor], list(batch))
            return torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=self.batch_first, padding_value=self.pad_value
            )
        if isinstance(batch[0], (list, tuple)):
            # tuple/list 样本按列 padding, 常见于 (tokens, labels) 这类变长序列.
            sequence_batch = cast(Sequence[Sequence[Any]], batch)
            transposed = list(zip(*sequence_batch))
            return tuple(
                torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(x) for x in col],
                    batch_first=self.batch_first,
                    padding_value=self.pad_value,
                )
                for col in transposed
            )
        return default_collate(list(batch))


@COLLATE_REGISTRY.register("DictCollate")
class DictCollate:
    """dict-batch collate, 每键独立 collate.

    Args:
        keys: 需要 collate 的键列表, None 表示全部键.
    """

    def __init__(self, keys: Optional[list[str]] = None) -> None:
        self.keys = keys

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(batch[0], dict):
            raise TypeError(f"DictCollate expects dict elements, got {type(batch[0])}")

        keys = self.keys if self.keys is not None else list(batch[0].keys())
        collated: Dict[str, Any] = {}
        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                first_shape = values[0].shape
                if all(isinstance(v, torch.Tensor) and v.shape == first_shape for v in values):
                    collated[key] = default_collate(values)
                else:
                    # 形状不一致的 tensor 保留为 list, 例如 pair/detection 的变长字段.
                    collated[key] = values
            else:
                collated[key] = list(values)
        return collated


class DetectionCollate:
    """Collate detection samples while preserving variable-sized targets."""

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        collated: Dict[str, Any] = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if key in {"boxes", "labels"}:
                # 检测目标数量逐图不同, 保留 list, 由 task/loss 决定后续处理方式.
                collated[key] = values
            elif torch.is_tensor(values[0]):
                collated[key] = default_collate(values)
            else:
                collated[key] = list(values)
        return collated


class ImageEditCollate:
    """Collate image edit samples into a dict batch."""

    def __init__(self, keys: Optional[Iterable[str]] = None) -> None:
        self.keys = list(keys) if keys is not None else None

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        keys = self.keys if self.keys is not None else list(batch[0].keys())
        collated: Dict[str, Any] = {}
        for key in keys:
            values = [item[key] for item in batch]
            if torch.is_tensor(values[0]):
                collated[key] = default_collate(values)
            else:
                collated[key] = list(values)
        return collated


__all__ = ["PadCollate", "DictCollate", "DetectionCollate", "ImageEditCollate"]
