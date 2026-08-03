"""Registry 系统测试。"""

import pytest

from xdl.errors import RegistryError
from xdl.utils.registry import (
    COLLATE_REGISTRY,
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    Registry,
)


class TestRegistryBasic:
    """注册表基本操作。"""

    def test_register_and_get(self):
        reg = Registry("test")
        reg.register("foo")(lambda: 42)
        assert reg.get("foo")() == 42

    def test_register_duplicate_warns(self):
        reg = Registry("test")
        reg.register("bar")(lambda: 1)
        # 第二次注册同名应跳过（不报错）
        result = reg.register("bar")(lambda: 2)
        assert result() == 2  # 返回原始对象
        assert reg.get("bar")() == 1  # 值未变

    def test_get_missing_raises(self):
        reg = Registry("test")
        with pytest.raises(RegistryError, match="not found"):
            reg.get("nonexistent")

    def test_list_available(self):
        reg = Registry("test")
        reg.register("a")(lambda: 0)
        reg.register("b")(lambda: 0)
        assert set(reg.list_available()) == {"a", "b"}


class TestGlobalRegistries:
    """全局注册表预注册验证。"""

    def test_model_registry_has_torch_nn(self):
        available = MODEL_REGISTRY.list_available()
        assert len(available) > 0

    def test_optimizer_registry_has_torch_optim(self):
        available = OPTIMIZER_REGISTRY.list_available()
        assert "Adam" in available
        assert "SGD" in available

    def test_loss_registry_has_torch_nn_loss(self):
        available = LOSS_REGISTRY.list_available()
        assert "CrossEntropyLoss" in available
        assert "MSELoss" in available
        # P0-2 新增
        for name in (
            "HuberLoss",
            "InfoNCE",
            "DiceLoss",
            "LabelSmoothingCrossEntropy",
            "SoftTargetCrossEntropy",
            "AsymmetricLoss",
            "CharbonnierLoss",
            "TotalVariationLoss",
            "GradientDifferenceLoss",
            "SSIMLoss",
            "ReconstructionLoss",
            "KLDivergenceLoss",
            "VAELoss",
            "GANLoss",
            "HingeDiscriminatorLoss",
            "HingeGeneratorLoss",
            "FeatureMatchingLoss",
            "DiffusionPredictionLoss",
            "JaccardLoss",
            "TverskyLoss",
            "FocalTverskyLoss",
            "DiceCrossEntropyLoss",
            "BoxIoULoss",
            "MaskedCrossEntropyLoss",
            "SequenceCrossEntropyLoss",
            "TokenClassificationLoss",
            "CausalLanguageModelingLoss",
        ):
            assert name in available, f"Missing {name}"

    def test_metric_registry_has_builtin(self):
        available = METRIC_REGISTRY.list_available()
        for name in (
            "Accuracy",
            "Precision",
            "Recall",
            "F1Score",
            "BalancedAccuracy",
            "MatthewsCorrCoef",
            "ConfusionMatrix",
            "MultiLabelAccuracy",
            "MultiLabelPrecision",
            "MultiLabelRecall",
            "MultiLabelF1Score",
            "PixelAccuracy",
            "MeanIoU",
            "FrequencyWeightedIoU",
            "BoxIoU",
            "MeanReciprocalRank",
            "HitRateAtK",
            "Perplexity",
            "TokenAccuracy",
            "SequenceExactMatch",
            "BLEUScore",
            "ROUGELScore",
            "PeakSignalNoiseRatio",
            "StructuralSimilarity",
            "MultiScaleStructuralSimilarity",
            "FrechetInceptionDistance",
            "KernelInceptionDistance",
            "InceptionScore",
            "PSNR",
            "SSIM",
            "MS_SSIM",
            "FID",
            "KID",
        ):
            assert name in available, f"Missing {name}"

    def test_dataset_registry_has_builtin(self):
        available = DATASET_REGISTRY.list_available()
        assert "SyntheticClassificationDataset" in available
        for name in (
            "ImageFolderDataset",
            "ImageTextSidecarDataset",
            "RecordRegressionDataset",
            "RecordMultiLabelClassificationDataset",
            "ImageMaskSidecarDataset",
            "RecordSegmentationDataset",
            "RecordDetectionDataset",
            "RecordTextDataset",
            "RecordPairDataset",
            "RecordTripletDataset",
            "ImageEditDataset",
        ):
            assert name in available, f"Missing {name}"

    def test_collate_registry_has_builtin(self):
        available = COLLATE_REGISTRY.list_available()
        assert "PadCollate" in available
        assert "DictCollate" in available
        assert "DetectionCollate" in available
        assert "ImageEditCollate" in available

    def test_scheduler_registry_empty_by_default(self):
        # scheduler 没有预注册 PyTorch 原生类
        assert isinstance(SCHEDULER_REGISTRY.list_available(), list)
