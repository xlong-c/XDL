"""评估指标测试。"""

import torch

from xdl.metric.metrics import (
    Accuracy,
    DiceCoefficient,
    F1Score,
    IoU,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    Recall,
    RootMeanSquaredError,
    TopKAccuracy,
)


class TestAccuracy:
    def test_multiclass(self, dummy_batch):
        acc = Accuracy(num_classes=10)(dummy_batch.pred, dummy_batch.target)
        assert 0.0 <= acc <= 1.0

    def test_binary(self):
        pred = torch.tensor([[0.8], [0.2], [0.9], [0.1]])
        target = torch.tensor([1, 0, 1, 0])
        acc = Accuracy(threshold=0.5)(pred, target)
        assert acc == 1.0

    def test_auto_detect(self):
        # pred dim=1 → 二分类
        acc = Accuracy()(torch.tensor([0.9, 0.1, 0.8, 0.2]), torch.tensor([1, 0, 1, 0]))
        assert 0.0 <= acc <= 1.0


class TestPrecisionRecallF1:
    def test_precision_macro(self, dummy_batch):
        p = Precision(num_classes=10, average="macro")(dummy_batch.pred, dummy_batch.target)
        assert 0.0 <= p <= 1.0

    def test_precision_micro(self, dummy_batch):
        p = Precision(num_classes=10, average="micro")(dummy_batch.pred, dummy_batch.target)
        assert 0.0 <= p <= 1.0

    def test_recall(self, dummy_batch):
        r = Recall(num_classes=10)(dummy_batch.pred, dummy_batch.target)
        assert 0.0 <= r <= 1.0

    def test_f1(self, dummy_batch):
        f1 = F1Score(num_classes=10)(dummy_batch.pred, dummy_batch.target)
        assert 0.0 <= f1 <= 1.0


class TestIoUDice:
    def test_iou_multiclass(self, dummy_image_batch):
        iou = IoU(num_classes=3, average="macro")(dummy_image_batch.pred, dummy_image_batch.target)
        assert 0.0 <= iou <= 1.0

    def test_dice_multiclass(self, dummy_image_batch):
        dice = DiceCoefficient(num_classes=3, average="macro")(
            dummy_image_batch.pred, dummy_image_batch.target
        )
        assert 0.0 <= dice <= 1.0

    def test_dice_micro(self, dummy_image_batch):
        dice = DiceCoefficient(num_classes=3, average="micro")(
            dummy_image_batch.pred, dummy_image_batch.target
        )
        assert 0.0 <= dice <= 1.0


class TestRegressionMetrics:
    def test_mae(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 3.0, 5.0])
        mae = MeanAbsoluteError()(pred, target)
        assert abs(mae - 1.0) < 1e-5

    def test_mse(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 3.0, 5.0])
        mse = MeanSquaredError()(pred, target)
        assert abs(mse - 5.0 / 3.0) < 1e-5

    def test_rmse(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 3.0, 5.0])
        rmse = RootMeanSquaredError()(pred, target)
        assert abs(rmse - (5.0 / 3.0) ** 0.5) < 1e-5


class TestTopKAccuracy:
    def test_top1(self):
        pred = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        target = torch.tensor([2, 0])
        acc = TopKAccuracy(k=1)(pred, target)
        assert acc == 1.0

    def test_top2(self):
        # 第0个: top2={2(0.7), 0(0.2)}, target=0 → 正确
        # 第1个: top2={0(0.8), 1(0.15)}, target=2 → 错误
        pred = torch.tensor([[0.2, 0.1, 0.7], [0.8, 0.15, 0.05]])
        target = torch.tensor([0, 2])
        acc = TopKAccuracy(k=2)(pred, target)
        assert acc == 0.5
