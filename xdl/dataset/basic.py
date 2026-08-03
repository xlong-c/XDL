"""
基础内置数据集。
"""

from typing import Sequence, Tuple

import torch
from torch.utils.data import Dataset


class SyntheticClassificationDataset(Dataset):
    """
    一个仅依赖 torch 的合成分类数据集，适合配置系统与训练链路 smoke test。
    """

    def __init__(
        self,
        num_samples: int = 32,
        input_shape: Sequence[int] = (4,),
        num_classes: int = 2,
        seed: int = 42,
    ):
        self.num_samples = int(num_samples)
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.num_classes = int(num_classes)

        generator = torch.Generator().manual_seed(int(seed))
        self.features = torch.randn((self.num_samples, *self.input_shape), generator=generator)
        self.targets = torch.randint(
            low=0,
            high=self.num_classes,
            size=(self.num_samples,),
            generator=generator,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]
