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
from .dense import (
    DetectionCollate,
    ImageBoxesTransform,
    ImageMaskSidecarDataset,
    ImageMaskTransform,
    ManifestDetectionDataset,
    ManifestSegmentationDataset,
    RecordDetectionDataset,
    RecordSegmentationDataset,
)
from .image_edit import (
    ImageEditCollate,
    ImageEditDataset,
    ManifestImageEditCollate,
    ManifestImageEditDataset,
    PairedImageTransform,
)
from .templates import (
    ImageFolderDataset,
    ImageFolderClassificationDataset,
    ImageTextSidecarDataset,
    ManifestClassificationDataset,
    ManifestDatasetBase,
    ManifestImageTextDataset,
    ManifestMultiLabelClassificationDataset,
    ManifestPairDataset,
    ManifestRegressionDataset,
    ManifestRecordDataset,
    ManifestTextDataset,
    ManifestTripletDataset,
    RecordClassificationDataset,
    RecordDataset,
    RecordDatasetBase,
    RecordImageTextDataset,
    RecordMultiLabelClassificationDataset,
    RecordPairDataset,
    RecordRegressionDataset,
    RecordTextDataset,
    RecordTripletDataset,
)

# 通用模板集中注册在这里, 让 YAML 可以统一使用 registry:Name 构建数据集.
register_dataset("SyntheticClassificationDataset")(SyntheticClassificationDataset)
register_dataset("ImageEditDataset")(ImageEditDataset)
register_dataset("ManifestImageEditDataset")(ImageEditDataset)
register_dataset("RecordDataset")(RecordDataset)
register_dataset("ManifestRecordDataset")(RecordDataset)
register_dataset("ImageFolderDataset")(ImageFolderDataset)
register_dataset("ImageFolderClassificationDataset")(ImageFolderClassificationDataset)
register_dataset("ImageTextSidecarDataset")(ImageTextSidecarDataset)
register_dataset("RecordClassificationDataset")(RecordClassificationDataset)
register_dataset("ManifestClassificationDataset")(RecordClassificationDataset)
register_dataset("RecordRegressionDataset")(RecordRegressionDataset)
register_dataset("ManifestRegressionDataset")(RecordRegressionDataset)
register_dataset("RecordMultiLabelClassificationDataset")(RecordMultiLabelClassificationDataset)
register_dataset("ManifestMultiLabelClassificationDataset")(RecordMultiLabelClassificationDataset)
register_dataset("ImageMaskSidecarDataset")(ImageMaskSidecarDataset)
register_dataset("RecordSegmentationDataset")(RecordSegmentationDataset)
register_dataset("ManifestSegmentationDataset")(RecordSegmentationDataset)
register_dataset("RecordDetectionDataset")(RecordDetectionDataset)
register_dataset("ManifestDetectionDataset")(RecordDetectionDataset)
register_dataset("RecordImageTextDataset")(RecordImageTextDataset)
register_dataset("ManifestImageTextDataset")(RecordImageTextDataset)
register_dataset("RecordTextDataset")(RecordTextDataset)
register_dataset("ManifestTextDataset")(RecordTextDataset)
register_dataset("RecordPairDataset")(RecordPairDataset)
register_dataset("ManifestPairDataset")(RecordPairDataset)
register_dataset("RecordTripletDataset")(RecordTripletDataset)
register_dataset("ManifestTripletDataset")(RecordTripletDataset)
register_transform("PairedImageTransform")(PairedImageTransform)
register_transform("ImageMaskTransform")(ImageMaskTransform)
register_transform("ImageBoxesTransform")(ImageBoxesTransform)
register_collate("ImageEditCollate")(ImageEditCollate)
register_collate("ManifestImageEditCollate")(ImageEditCollate)
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
    "ImageEditDataset",
    "ManifestImageEditDataset",
    "PairedImageTransform",
    "ImageEditCollate",
    "ManifestImageEditCollate",
    "RecordDataset",
    "ManifestRecordDataset",
    "ImageFolderDataset",
    "ImageFolderClassificationDataset",
    "ImageTextSidecarDataset",
    "ImageMaskSidecarDataset",
    "RecordDatasetBase",
    "ManifestDatasetBase",
    "RecordClassificationDataset",
    "ManifestClassificationDataset",
    "RecordRegressionDataset",
    "ManifestRegressionDataset",
    "RecordMultiLabelClassificationDataset",
    "ManifestMultiLabelClassificationDataset",
    "RecordSegmentationDataset",
    "ManifestSegmentationDataset",
    "RecordDetectionDataset",
    "ManifestDetectionDataset",
    "RecordImageTextDataset",
    "ManifestImageTextDataset",
    "RecordTextDataset",
    "ManifestTextDataset",
    "RecordPairDataset",
    "ManifestPairDataset",
    "RecordTripletDataset",
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
