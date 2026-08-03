"""数据集测试。"""

import torch

from xdl.dataset.basic import SyntheticClassificationDataset


class TestSyntheticDataset:
    def test_length(self):
        ds = SyntheticClassificationDataset(num_samples=20, input_shape=[10], num_classes=5)
        assert len(ds) == 20

    def test_item_shape(self):
        ds = SyntheticClassificationDataset(num_samples=10, input_shape=[3, 32, 32], num_classes=4)
        feature, target = ds[0]
        assert tuple(feature.shape) == (3, 32, 32)
        assert target.numel() == 1
        assert 0 <= target.item() < 4

    def test_reproducibility(self):
        ds1 = SyntheticClassificationDataset(num_samples=5, input_shape=[8], num_classes=3, seed=42)
        ds2 = SyntheticClassificationDataset(num_samples=5, input_shape=[8], num_classes=3, seed=42)
        for i in range(5):
            assert torch.equal(ds1[i][0], ds2[i][0])
            assert ds1[i][1] == ds2[i][1]
