"""损失函数测试。"""

import torch

from xdl.loss.contrastive_loss import InfoNCE
from xdl.loss.dice_loss import DiceLoss, GeneralizedDiceLoss
from xdl.loss.focal_loss import BinaryFocalLoss, FocalLoss
from xdl.loss.huber_loss import HuberLoss


class TestHuberLoss:
    def test_shape(self):
        loss_fn = HuberLoss(delta=1.0)
        out = loss_fn(torch.randn(4, 10), torch.randn(4, 10))
        assert out.ndim == 0  # scalar

    def test_gradient(self):
        loss_fn = HuberLoss()
        pred = torch.randn(4, 10, requires_grad=True)
        target = torch.randn(4, 10)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_delta_behavior(self):
        """小误差用 MSE, 大误差用 MAE。"""
        loss_fn = HuberLoss(delta=1.0)
        # 误差 = 0 → loss = 0
        small = loss_fn(torch.zeros(4), torch.zeros(4))
        assert small.item() == 0.0


class TestFocalLoss:
    def test_shape(self):
        loss_fn = FocalLoss()
        pred = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        out = loss_fn(pred, target)
        assert out.ndim == 0

    def test_gradient(self):
        loss_fn = FocalLoss()
        pred = torch.randn(4, 10, requires_grad=True)
        target = torch.randint(0, 10, (4,))
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_binary(self):
        loss_fn = BinaryFocalLoss()
        pred = torch.randn(8, requires_grad=True)
        target = torch.randint(0, 2, (8,)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0


class TestContrastiveLoss:
    def test_info_nce(self):
        loss_fn = InfoNCE(temperature=0.1)
        features = torch.randn(8, 64)  # 4 anchor + 4 positive
        out = loss_fn(features)
        assert out.ndim == 0
        assert out.item() > 0

    def test_gradient(self):
        loss_fn = InfoNCE()
        features = torch.randn(8, 64, requires_grad=True)
        loss = loss_fn(features)
        loss.backward()
        assert features.grad is not None


class TestDiceLoss:
    def test_multiclass(self):
        loss_fn = DiceLoss(smooth=1.0, average="macro")
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 3, (2, 32, 32))
        out = loss_fn(pred, target)
        assert 0.0 <= out.item() <= 1.0

    def test_micro(self):
        loss_fn = DiceLoss(average="micro")
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 3, (2, 32, 32))
        out = loss_fn(pred, target)
        assert 0.0 <= out.item() <= 1.0

    def test_gradient(self):
        loss_fn = DiceLoss()
        pred = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randint(0, 3, (2, 32, 32))
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_generalized(self):
        loss_fn = GeneralizedDiceLoss(smooth=1.0)
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 3, (2, 32, 32))
        out = loss_fn(pred, target)
        assert 0.0 <= out.item() <= 1.0
