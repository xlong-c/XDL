"""
数据集注册入口。

注意：部分数据集依赖可选三方库，导入失败时不会阻断整个框架。
"""

from xdl.utils.registry import register_collate, register_dataset, register_transform

_import_errors = {}
GridImageCsvDataset = None
GridImageDirDataset = None
Hair10HairDataset = None

from . import collate  # 注册 PadCollate / DictCollate
from .basic import SyntheticClassificationDataset
from .manifest_dense import (
    DetectionCollate,
    ImageBoxesTransform,
    ImageMaskTransform,
    ManifestDetectionDataset,
    ManifestSegmentationDataset,
)
from .manifest_image_edit import (
    ManifestImageEditCollate,
    ManifestImageEditDataset,
    PairedImageTransform,
)
from .templates import (
    ImageFolderDataset,
    ImageFolderClassificationDataset,
    ImageTextSidecarDataset,
    ManifestClassificationDataset,
    ManifestImageTextDataset,
    ManifestMultiLabelClassificationDataset,
    ManifestPairDataset,
    ManifestRegressionDataset,
    ManifestRecordDataset,
    ManifestTextDataset,
    ManifestTripletDataset,
)

register_dataset("SyntheticClassificationDataset")(SyntheticClassificationDataset)
register_dataset("ManifestImageEditDataset")(ManifestImageEditDataset)
register_dataset("ManifestRecordDataset")(ManifestRecordDataset)
register_dataset("ImageFolderDataset")(ImageFolderDataset)
register_dataset("ImageFolderClassificationDataset")(ImageFolderClassificationDataset)
register_dataset("ImageTextSidecarDataset")(ImageTextSidecarDataset)
register_dataset("ManifestClassificationDataset")(ManifestClassificationDataset)
register_dataset("ManifestRegressionDataset")(ManifestRegressionDataset)
register_dataset("ManifestMultiLabelClassificationDataset")(ManifestMultiLabelClassificationDataset)
register_dataset("ManifestSegmentationDataset")(ManifestSegmentationDataset)
register_dataset("ManifestDetectionDataset")(ManifestDetectionDataset)
register_dataset("ManifestImageTextDataset")(ManifestImageTextDataset)
register_dataset("ManifestTextDataset")(ManifestTextDataset)
register_dataset("ManifestPairDataset")(ManifestPairDataset)
register_dataset("ManifestTripletDataset")(ManifestTripletDataset)
register_transform("PairedImageTransform")(PairedImageTransform)
register_transform("ImageMaskTransform")(ImageMaskTransform)
register_transform("ImageBoxesTransform")(ImageBoxesTransform)
register_collate("ManifestImageEditCollate")(ManifestImageEditCollate)
register_collate("DetectionCollate")(DetectionCollate)

# 视觉数据集
from .vision_datasets import CIFAR10Dataset, MNISTDataset

if CIFAR10Dataset is not None:
    register_dataset("CIFAR10")(CIFAR10Dataset)
if MNISTDataset is not None:
    register_dataset("MNIST")(MNISTDataset)

try:
    from .hairdata import GridImageDataset as GridImageCsvDataset

    register_dataset("GridImageCsvDataset")(GridImageCsvDataset)
except Exception as exc:  # pragma: no cover - 依赖缺失时允许跳过
    _import_errors["GridImageCsvDataset"] = exc

try:
    from .hairdata3y import GridImageDataset as GridImageDirDataset

    register_dataset("GridImageDirDataset")(GridImageDirDataset)
except Exception as exc:  # pragma: no cover - 依赖缺失时允许跳过
    _import_errors["GridImageDirDataset"] = exc

try:
    from .hairdata10hair import Hair10HairDataset

    register_dataset("Hair10HairDataset")(Hair10HairDataset)
except Exception as exc:  # pragma: no cover - 依赖缺失时允许跳过
    _import_errors["Hair10HairDataset"] = exc

__all__ = [
    "SyntheticClassificationDataset",
    "ManifestImageEditDataset",
    "PairedImageTransform",
    "ManifestImageEditCollate",
    "ManifestRecordDataset",
    "ImageFolderDataset",
    "ImageFolderClassificationDataset",
    "ImageTextSidecarDataset",
    "ManifestClassificationDataset",
    "ManifestRegressionDataset",
    "ManifestMultiLabelClassificationDataset",
    "ManifestSegmentationDataset",
    "ManifestDetectionDataset",
    "ManifestImageTextDataset",
    "ManifestTextDataset",
    "ManifestPairDataset",
    "ManifestTripletDataset",
    "GridImageCsvDataset",
    "GridImageDirDataset",
    "Hair10HairDataset",
    "CIFAR10Dataset",
    "MNISTDataset",
    "ImageMaskTransform",
    "ImageBoxesTransform",
    "DetectionCollate",
]
