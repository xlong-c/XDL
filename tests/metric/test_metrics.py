"""评估指标测试。"""

import torch

from xdl.metric.image_generation import (
    FrechetInceptionDistance,
    InceptionScore,
    KernelInceptionDistance,
    MultiScaleStructuralSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarity,
)
from xdl.metric.metrics import (
    Accuracy,
    BLEUScore,
    BalancedAccuracy,
    BoxIoU,
    ConfusionMatrix,
    DetectionMeanAveragePrecision,
    DiceCoefficient,
    F1Score,
    FrequencyWeightedIoU,
    HitRateAtK,
    IoU,
    MatthewsCorrCoef,
    MeanAbsoluteError,
    MeanIoU,
    MeanReciprocalRank,
    MeanSquaredError,
    MultiLabelAccuracy,
    MultiLabelF1Score,
    MultiLabelPrecision,
    MultiLabelRecall,
    Perplexity,
    PixelAccuracy,
    Precision,
    ROUGELScore,
    Recall,
    RootMeanSquaredError,
    SequenceExactMatch,
    TopKAccuracy,
    TokenAccuracy,
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
        p = Precision(num_classes=10, average="macro")(
            dummy_batch.pred, dummy_batch.target
        )
        assert 0.0 <= p <= 1.0

    def test_precision_micro(self, dummy_batch):
        p = Precision(num_classes=10, average="micro")(
            dummy_batch.pred, dummy_batch.target
        )
        assert 0.0 <= p <= 1.0

    def test_recall(self, dummy_batch):
        r = Recall(num_classes=10)(dummy_batch.pred, dummy_batch.target)
        assert 0.0 <= r <= 1.0

    def test_f1(self, dummy_batch):
        f1 = F1Score(num_classes=10)(dummy_batch.pred, dummy_batch.target)
        assert 0.0 <= f1 <= 1.0


class TestIoUDice:
    def test_iou_multiclass(self, dummy_image_batch):
        iou = IoU(num_classes=3, average="macro")(
            dummy_image_batch.pred, dummy_image_batch.target
        )
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


class TestImageReconstructionMetrics:
    def test_psnr_identical_images_is_large(self):
        image = torch.rand(2, 3, 8, 8)
        psnr = PeakSignalNoiseRatio()(image, image)
        assert psnr > 90.0

    def test_structural_similarity_identical_images(self):
        image = torch.rand(2, 3, 8, 8)
        ssim = StructuralSimilarity(window_size=3)(image, image)
        assert abs(ssim - 1.0) < 1e-5

    def test_multiscale_structural_similarity(self):
        image = torch.rand(2, 3, 16, 16)
        degraded = (image * 0.9).clamp(0, 1)
        ms_ssim = MultiScaleStructuralSimilarity(window_size=3, levels=2)(
            image, degraded
        )
        assert 0.0 <= ms_ssim <= 1.0


class TestImageGenerationMetrics:
    def test_fid_identical_features_is_near_zero(self):
        features = torch.randn(8, 4)
        fid = FrechetInceptionDistance()(features, features)
        assert fid < 1e-4

    def test_kid_identical_shape(self):
        real = torch.randn(8, 4)
        fake = real + 0.01 * torch.randn(8, 4)
        kid = KernelInceptionDistance()(fake, real)
        assert isinstance(kid, float)

    def test_inception_score_from_logits(self):
        logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        score = InceptionScore(splits=1)(logits)
        assert score > 1.0


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


class TestClassificationExpansionMetrics:
    def test_confusion_matrix_accumulates_and_resets(self):
        metric = ConfusionMatrix(num_classes=3)
        pred = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([0, 1, 2])
        matrix = metric(pred, target)
        assert matrix.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
        metric.reset()
        assert metric.compute().sum().item() == 0

    def test_balanced_accuracy_stateful(self):
        metric = BalancedAccuracy(num_classes=2)
        metric.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        metric.update(torch.tensor([1, 1]), torch.tensor([0, 1]))
        assert abs(metric.compute() - 0.75) < 1e-6

    def test_balanced_accuracy_accepts_label_predictions(self):
        metric = BalancedAccuracy(num_classes=3)
        pred = torch.tensor([0, 2, 2, 1])
        target = torch.tensor([0, 1, 2, 1])
        assert abs(metric(pred, target) - ((1.0 + 0.5 + 1.0) / 3.0)) < 1e-6

    def test_matthews_corrcoef_binary(self):
        metric = MatthewsCorrCoef(num_classes=2)
        value = metric(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1, 1]))
        assert -1.0 <= value <= 1.0


class TestMultiLabelMetrics:
    def test_multilabel_metrics_micro(self):
        pred = torch.tensor([[0.9, 0.2, 0.8], [0.1, 0.7, 0.2]])
        target = torch.tensor([[1, 0, 1], [0, 1, 0]])
        assert MultiLabelAccuracy(from_logits=False)(pred, target) == 1.0
        assert MultiLabelPrecision(from_logits=False)(pred, target) == 1.0
        assert MultiLabelRecall(from_logits=False)(pred, target) == 1.0
        assert MultiLabelF1Score(from_logits=False)(pred, target) == 1.0

    def test_multilabel_macro_penalizes_false_positive(self):
        pred = torch.tensor([[0.9, 0.9], [0.9, 0.1]])
        target = torch.tensor([[1, 0], [1, 0]])
        f1 = MultiLabelF1Score(average="macro", from_logits=False)(pred, target)
        assert 0.0 <= f1 < 1.0


class TestSegmentationExpansionMetrics:
    def test_pixel_accuracy_with_ignore_index(self):
        pred = torch.tensor([[[[4.0, 0.0], [0.0, 3.0]], [[0.0, 5.0], [2.0, 0.0]]]])
        target = torch.tensor([[[0, 1], [255, 1]]])
        acc = PixelAccuracy(ignore_index=255)(pred, target)
        assert abs(acc - (2.0 / 3.0)) < 1e-6

    def test_mean_iou_stateful(self):
        metric = MeanIoU(num_classes=2)
        pred = torch.tensor([[[0, 1], [1, 0]]])
        target = torch.tensor([[[0, 1], [0, 0]]])
        metric.update(pred, target)
        assert 0.0 <= metric.compute() <= 1.0
        assert abs(metric.compute() - ((2.0 / 3.0 + 1.0 / 2.0) / 2.0)) < 1e-6

    def test_pixel_accuracy_accepts_label_map_predictions(self):
        pred = torch.tensor([[[0, 1], [1, 2]]])
        target = torch.tensor([[[0, 2], [1, 2]]])
        acc = PixelAccuracy()(pred, target)
        assert acc == 0.75

    def test_frequency_weighted_iou(self):
        metric = FrequencyWeightedIoU(num_classes=2)
        pred = torch.tensor([[[0, 1], [1, 0]]])
        target = torch.tensor([[[0, 1], [0, 0]]])
        value = metric(pred, target)
        assert 0.0 <= value <= 1.0


class TestDetectionMetrics:
    def test_box_iou(self):
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
        target = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]])
        iou = BoxIoU(reduction="none")(pred, target)
        assert torch.allclose(iou, torch.tensor([1.0, 0.0]))

    def test_detection_mean_average_precision(self):
        predictions = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                "labels": torch.tensor([1]),
            }
        ]

        metrics = DetectionMeanAveragePrecision(iou_thresholds=[0.5, 0.75])(
            predictions,
            targets,
        )

        assert metrics["map50_95"] == 1.0
        assert metrics["map50"] == 1.0
        assert metrics["map75"] == 1.0


class TestNLPAndRetrievalMetrics:
    def test_perplexity_and_token_accuracy_ignore_index(self):
        logits = torch.tensor(
            [
                [[5.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
                [[0.0, 0.0, 5.0], [3.0, 0.0, 0.0]],
            ]
        )
        target = torch.tensor([[0, 1], [2, -100]])
        assert Perplexity()(logits, target) >= 1.0
        assert TokenAccuracy()(logits, target) == 1.0

    def test_sequence_exact_match(self):
        pred = torch.tensor([[1, 2, 3], [1, 2, 0]])
        target = torch.tensor([[1, 2, 3], [1, 2, 3]])
        assert SequenceExactMatch()(pred, target) == 0.5

    def test_retrieval_metrics(self):
        scores = torch.tensor([[0.1, 0.9, 0.2], [0.3, 0.2, 0.1]])
        target = torch.tensor([1, 2])
        assert (
            abs(MeanReciprocalRank()(scores, target) - (1.0 + 1.0 / 3.0) / 2.0) < 1e-6
        )
        assert HitRateAtK(k=2)(scores, target) == 0.5

    def test_text_generation_metrics(self):
        pred = ["the cat sat", "a short summary"]
        target = ["the cat sat", "a summary"]
        bleu = BLEUScore(max_n=2)(pred, target)
        rouge = ROUGELScore()(pred, target)
        assert 0.0 <= bleu <= 1.0
        assert 0.0 <= rouge <= 1.0
        assert ROUGELScore()(["same tokens"], ["same tokens"]) == 1.0
