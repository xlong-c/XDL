"""内置 collate 函数。"""

import torch
from xdl.utils.registry import COLLATE_REGISTRY
from torch.utils.data.dataloader import default_collate
@COLLATE_REGISTRY.register("PadCollate")
class PadCollate:
    """不等长序列 collate — pad 到批次内最大长度。

    Args:
        pad_value: 填充值, 默认 0。
        batch_first: 输出 batch 维度是否在前, 默认 True。
    """

    def __init__(self, pad_value: float = 0.0, batch_first: bool = True):
        self.pad_value = pad_value
        self.batch_first = batch_first

    def __call__(self, batch):
        if isinstance(batch[0], torch.Tensor):
            return torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=self.batch_first, padding_value=self.pad_value
            )
        if isinstance(batch[0], (list, tuple)):
            transposed = list(zip(*batch))
            return tuple(
                torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(x) for x in col],
                    batch_first=self.batch_first,
                    padding_value=self.pad_value,
                )
                for col in transposed
            )
        return default_collate(batch)


@COLLATE_REGISTRY.register("DictCollate")
class DictCollate:
    """dict-batch collate, 每键独立 collate。

    Args:
        keys: 需要 collate 的键列表, None 表示全部键。
    """

    def __init__(self, keys: list[str] | None = None):
        self.keys = keys

    def __call__(self, batch):
        if not isinstance(batch[0], dict):
            raise TypeError(f"DictCollate expects dict elements, got {type(batch[0])}")

        keys = self.keys if self.keys is not None else list(batch[0].keys())
        collated = {}
        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = default_collate(values)
                except RuntimeError:
                    collated[key] = values
            else:
                collated[key] = list(values)
        return collated
