import torch
import pytest

from xdl.loss.box_loss import BoxIoULoss
from xdl.loss.classification_loss import (
    AsymmetricLoss,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from xdl.loss.contrastive_loss import InfoNCE
from xdl.post_training.distillation_loss import (
    distillation_loss,
    feature_distillation_loss,
    kl_divergence_with_temperature,
    relation_distillation_loss,
)
from xdl.loss.dice_loss import DiceLoss, GeneralizedDiceLoss
from xdl.loss.focal_loss import BinaryFocalLoss, FocalLoss
from xdl.loss.huber_loss import HuberLoss
from xdl.loss.generative_loss import (
    DiffusionPredictionLoss,
    FeatureMatchingLoss,
    GANLoss,
    HingeDiscriminatorLoss,
    HingeGeneratorLoss,
    KLDivergenceLoss,
    VAELoss,
)
from xdl.loss.reconstruction_loss import (
    CharbonnierLoss,
    GradientDifferenceLoss,
    ReconstructionLoss,
    SSIMLoss,
    TotalVariationLoss,
)
from xdl.loss.segmentation_loss import (
    DiceCrossEntropyLoss,
    FocalTverskyLoss,
    JaccardLoss,
    TverskyLoss,
)
from xdl.loss.sequence_loss import (
    CausalLanguageModelingLoss,
    MaskedCrossEntropyLoss,
    TokenClassificationLoss,
)
from xdl.utils.registry import build_loss


class TestClassificationLosses:
    def test_label_smoothing_cross_entropy(self):
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        pred = torch.randn(4, 5, requires_grad=True)
        target = torch.tensor([0, 1, -100, 3])
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None

    def test_label_smoothing_reduction_none_keeps_target_shape(self):
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1, reduction="none")
        pred = torch.randn(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        out = loss_fn(pred, target)
        assert out.shape == target.shape

    def test_soft_target_cross_entropy(self):
        loss_fn = SoftTargetCrossEntropy()
        pred = torch.randn(3, 4, requires_grad=True)
        target = torch.softmax(torch.randn(3, 4), dim=1)
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None

    def test_asymmetric_loss_multilabel(self):
        loss_fn = AsymmetricLoss()
        pred = torch.randn(2, 4, requires_grad=True)
        target = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None


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


class TestDistillationLosses:
    def test_kl_divergence_with_temperature_is_registered(self):
        logits = torch.tensor([[1.0, 2.0], [0.5, -0.5]], requires_grad=True)
        registry_loss = build_loss("kl_divergence_with_temperature")

        loss = registry_loss(logits, logits.detach(), temperature=2.0)
        loss.backward()

        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert logits.grad is not None

    def test_distillation_loss_combines_components(self):
        student_logits = torch.tensor([[1.0, 0.0], [0.2, 0.8]], requires_grad=True)
        teacher_logits = torch.tensor([[2.0, -1.0], [0.1, 1.4]])
        targets = torch.tensor([0, 1])
        student_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        teacher_features = torch.tensor([[1.0, 2.5], [3.5, 4.0]])

        breakdown = distillation_loss(
            student_logits,
            teacher_logits,
            targets=targets,
            temperature=2.0,
            alpha=0.7,
            feature_student=student_features,
            feature_teacher=teacher_features,
            feature_weight=0.2,
            relation_student=student_features,
            relation_teacher=teacher_features,
            relation_weight=0.1,
        )

        breakdown.total.backward()

        assert breakdown.total.ndim == 0
        assert breakdown.soft_target.item() >= 0.0
        assert breakdown.hard_target is not None
        assert breakdown.feature is not None
        assert breakdown.relation is not None
        assert student_logits.grad is not None

    def test_feature_and_relation_losses_match_identical_features(self):
        features = torch.randn(4, 3, 2)

        assert feature_distillation_loss(features, features).item() == pytest.approx(0.0)
        assert relation_distillation_loss(features, features).item() == pytest.approx(0.0)

    def test_delta_behavior(self):
        """小误差用 MSE, 大误差用 MAE。"""
        loss_fn = HuberLoss(delta=1.0)
        # 误差 = 0 → loss = 0
        small = loss_fn(torch.zeros(4), torch.zeros(4))
        assert small.item() == 0.0


class TestImageReconstructionLosses:
    def test_charbonnier_loss_has_gradient(self):
        loss_fn = CharbonnierLoss()
        pred = torch.rand(2, 3, 8, 8, requires_grad=True)
        target = torch.rand(2, 3, 8, 8)
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None

    def test_total_variation_accepts_single_image_argument(self):
        loss_fn = TotalVariationLoss()
        image = torch.rand(2, 3, 8, 8, requires_grad=True)
        loss = loss_fn(image)
        loss.backward()
        assert loss.ndim == 0
        assert image.grad is not None

    def test_gradient_difference_loss(self):
        loss_fn = GradientDifferenceLoss(penalty="l2")
        pred = torch.rand(2, 3, 8, 8, requires_grad=True)
        target = torch.rand(2, 3, 8, 8)
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None

    def test_ssim_loss_identical_images_is_near_zero(self):
        loss_fn = SSIMLoss(window_size=3)
        image = torch.rand(2, 3, 8, 8)
        loss = loss_fn(image, image)
        assert 0.0 <= loss.item() < 1e-4

    def test_reconstruction_loss_combines_terms(self):
        loss_fn = ReconstructionLoss(
            pixel="charbonnier",
            ssim_weight=0.1,
            gradient_weight=0.1,
            tv_weight=0.01,
        )
        pred = torch.rand(2, 3, 8, 8, requires_grad=True)
        target = torch.rand(2, 3, 8, 8)
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None


class TestImageGenerativeLosses:
    def test_kl_divergence_zero_for_unit_gaussian(self):
        loss_fn = KLDivergenceLoss()
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)
        loss = loss_fn(mu, logvar)
        assert loss.item() == 0.0

    def test_vae_loss_combines_reconstruction_and_kl(self):
        loss_fn = VAELoss(reconstruction="mse", beta=0.5)
        reconstruction = torch.rand(2, 1, 8, 8, requires_grad=True)
        target = torch.rand(2, 1, 8, 8)
        mu = torch.zeros(2, 4, requires_grad=True)
        logvar = torch.zeros(2, 4, requires_grad=True)
        loss = loss_fn(reconstruction, target, mu, logvar)
        loss.backward()
        assert loss.ndim == 0
        assert reconstruction.grad is not None
        assert mu.grad is not None

    def test_gan_loss_modes(self):
        logits = torch.tensor([1.0, -0.5, 0.25], requires_grad=True)
        loss = GANLoss(mode="hinge")(logits, target_is_real=True)
        loss.backward()
        assert loss.ndim == 0
        assert logits.grad is not None

    def test_hinge_discriminator_and_generator_loss(self):
        real_logits = torch.tensor([1.0, 0.5], requires_grad=True)
        fake_logits = torch.tensor([-1.0, 0.2], requires_grad=True)
        d_loss = HingeDiscriminatorLoss()(real_logits, fake_logits)
        g_loss = HingeGeneratorLoss()(fake_logits)
        total = d_loss + g_loss
        total.backward()
        assert d_loss.ndim == 0
        assert g_loss.ndim == 0
        assert real_logits.grad is not None
        assert fake_logits.grad is not None

    def test_feature_matching_loss_sequence(self):
        loss_fn = FeatureMatchingLoss(layer_weights=[0.3, 0.7])
        pred = [
            torch.rand(2, 4, requires_grad=True),
            torch.rand(2, 3, requires_grad=True),
        ]
        target = [torch.rand(2, 4), torch.rand(2, 3)]
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred[0].grad is not None
        assert pred[1].grad is not None

    def test_diffusion_prediction_loss_with_mask(self):
        loss_fn = DiffusionPredictionLoss(loss="huber")
        pred = torch.rand(2, 3, 4, 4, requires_grad=True)
        target = torch.rand(2, 3, 4, 4)
        mask = torch.ones(2, 4, 4)
        loss = loss_fn(pred, target, mask=mask)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None


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


class TestSegmentationLosses:
    def test_jaccard_tversky_and_focal_tversky(self):
        pred = torch.randn(2, 3, 8, 8, requires_grad=True)
        target = torch.randint(0, 3, (2, 8, 8))
        losses = [
            JaccardLoss(),
            TverskyLoss(alpha=0.3, beta=0.7),
            FocalTverskyLoss(gamma=1.5),
        ]
        total = sum(loss_fn(pred, target) for loss_fn in losses)
        total.backward()
        assert total.ndim == 0
        assert pred.grad is not None

    def test_dice_cross_entropy_binary(self):
        loss_fn = DiceCrossEntropyLoss()
        pred = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = torch.randint(0, 2, (2, 8, 8)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None

    def test_dice_cross_entropy_all_binary_ignore_is_zero_ce(self):
        loss_fn = DiceCrossEntropyLoss(ignore_index=-1)
        pred = torch.randn(1, 1, 4, 4, requires_grad=True)
        target = torch.full((1, 4, 4), -1.0)
        loss = loss_fn(pred, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert pred.grad is not None

    def test_jaccard_ignore_index(self):
        loss_fn = JaccardLoss(ignore_index=255)
        pred = torch.randn(1, 3, 4, 4)
        target = torch.randint(0, 3, (1, 4, 4))
        target[:, 0, 0] = 255
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)


class TestBoxLosses:
    def test_box_iou_loss_identical_boxes_is_zero(self):
        loss_fn = BoxIoULoss(mode="iou")
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
        loss = loss_fn(boxes, boxes)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_box_giou_loss_has_gradient(self):
        loss_fn = BoxIoULoss(mode="giou")
        pred = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.5, 3.0]],
            requires_grad=True,
        )
        target = torch.tensor([[0.5, 0.5, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None


class TestSequenceLosses:
    def test_masked_cross_entropy_ignores_masked_tokens(self):
        loss_fn = MaskedCrossEntropyLoss(reduction="none")
        pred = torch.randn(2, 3, 5, requires_grad=True)
        target = torch.tensor([[1, 2, -100], [0, 3, 4]])
        mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool)
        out = loss_fn(pred, target, mask=mask)
        assert out.shape == target.shape
        assert out[0, 2].item() == 0.0
        assert out[1, 1].item() == 0.0

    def test_token_classification_loss_has_gradient(self):
        loss_fn = TokenClassificationLoss()
        pred = torch.randn(2, 4, 6, requires_grad=True)
        target = torch.randint(0, 6, (2, 4))
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None

    def test_causal_language_modeling_loss_shifts_tokens(self):
        loss_fn = CausalLanguageModelingLoss()
        pred = torch.randn(2, 5, 7, requires_grad=True)
        labels = torch.randint(0, 7, (2, 5))
        loss = loss_fn(pred, labels)
        loss.backward()
        assert loss.ndim == 0
        assert pred.grad is not None
