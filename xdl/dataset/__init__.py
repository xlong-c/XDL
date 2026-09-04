"""
数据集注册入口.

注意: 部分数据集依赖可选三方库, 导入失败时不会阻断整个框架.
"""

from xdl.utils.registry import register_collate, register_dataset, register_transform

_import_errors = {}
GridImageCsvDataset = None
GridImageDirDataset = None
Hair10HairDataset = None

from . import collate  # 注册 PadCollate / DictCollate
from .basic import SyntheticClassificationDataset
from .classification import (
    ImageFolderClassificationDataset,
    RecordClassificationDataset,
    RecordMultiLabelClassificationDataset,
)
from .collate import DetectionCollate, DictCollate, ImageEditCollate, PadCollate
from .detection import RecordDetectionDataset
from .folder import ImageFolderDataset
from .image_edit import ImageEditDataset
from .image_text import ImagePromptDataset, ImageTextSidecarDataset, RecordImageTextDataset
from .pair import RecordPairDataset
from .record import RecordDataset, RecordDatasetBase
from .regression import RecordRegressionDataset
from .segmentation import ImageMaskSidecarDataset, RecordSegmentationDataset
from .sidecar import BasenameAlignedDataset
from .split import split_dataset, train_val_split
from .text import RecordTextDataset
from .transforms import (
    ImageBoxesTransform,
    ImageMaskTransform,
    PairedImageTransform,
)
from .triplet import RecordTripletDataset


# 通用模板集中注册在这里, 让 YAML 可以统一使用 registry:Name 构建数据集.
register_dataset("SyntheticClassificationDataset")(SyntheticClassificationDataset)
register_dataset("ImageEditDataset")(ImageEditDataset)
register_dataset("BasenameAlignedDataset")(BasenameAlignedDataset)
register_dataset("RecordDataset")(RecordDataset)
register_dataset("ImageFolderDataset")(ImageFolderDataset)
register_dataset("ImageFolderClassificationDataset")(ImageFolderClassificationDataset)
register_dataset("ImagePromptDataset")(ImagePromptDataset)
register_dataset("ImageTextSidecarDataset")(ImageTextSidecarDataset)
register_dataset("RecordClassificationDataset")(RecordClassificationDataset)
register_dataset("RecordRegressionDataset")(RecordRegressionDataset)
register_dataset("RecordMultiLabelClassificationDataset")(
    RecordMultiLabelClassificationDataset
)
register_dataset("ImageMaskSidecarDataset")(ImageMaskSidecarDataset)
register_dataset("RecordSegmentationDataset")(RecordSegmentationDataset)
register_dataset("RecordDetectionDataset")(RecordDetectionDataset)
register_dataset("RecordImageTextDataset")(RecordImageTextDataset)
register_dataset("RecordTextDataset")(RecordTextDataset)
register_dataset("RecordPairDataset")(RecordPairDataset)
register_dataset("RecordTripletDataset")(RecordTripletDataset)
# 历史 Manifest* registry 键保留为兼容别名. 新配置优先使用 Record* / ImageEdit* 名称.
register_dataset("ManifestRegressionDataset")(RecordRegressionDataset)
register_dataset("ManifestSegmentationDataset")(RecordSegmentationDataset)
register_dataset("ManifestDetectionDataset")(RecordDetectionDataset)
register_dataset("ManifestPairDataset")(RecordPairDataset)
register_dataset("ManifestImageEditDataset")(ImageEditDataset)
register_transform("PairedImageTransform")(PairedImageTransform)
register_transform("ImageMaskTransform")(ImageMaskTransform)
register_transform("ImageBoxesTransform")(ImageBoxesTransform)
register_collate("ImageEditCollate")(ImageEditCollate)
register_collate("ManifestImageEditCollate")(ImageEditCollate)
register_collate("DetectionCollate")(DetectionCollate)

# 视觉数据集
from .vision import CIFAR10Dataset, MNISTDataset

if CIFAR10Dataset is not None:
    register_dataset("CIFAR10")(CIFAR10Dataset)
if MNISTDataset is not None:
    register_dataset("MNIST")(MNISTDataset)

try:
    from .hair.hairdata import GridImageDataset as GridImageCsvDataset

    register_dataset("GridImageCsvDataset")(GridImageCsvDataset)
except Exception as exc:  # pragma: no cover - 依赖缺失时允许跳过
    _import_errors["GridImageCsvDataset"] = exc

try:
    from .hair.hairdata3y import GridImageDataset as GridImageDirDataset

    register_dataset("GridImageDirDataset")(GridImageDirDataset)
except Exception as exc:  # pragma: no cover - 依赖缺失时允许跳过
    _import_errors["GridImageDirDataset"] = exc

try:
    from .hair.hairdata10hair import Hair10HairDataset

    register_dataset("Hair10HairDataset")(Hair10HairDataset)
except Exception as exc:  # pragma: no cover - 依赖缺失时允许跳过
    _import_errors["Hair10HairDataset"] = exc

__all__ = [
    "SyntheticClassificationDataset",
    "ImageEditDataset",
    "BasenameAlignedDataset",
    "PairedImageTransform",
    "PadCollate",
    "DictCollate",
    "ImageEditCollate",
    "RecordDataset",
    "ImageFolderDataset",
    "ImageFolderClassificationDataset",
    "ImageTextSidecarDataset",
    "ImagePromptDataset",
    "ImageMaskSidecarDataset",
    "RecordDatasetBase",
    "RecordClassificationDataset",
    "RecordRegressionDataset",
    "RecordMultiLabelClassificationDataset",
    "RecordSegmentationDataset",
    "RecordDetectionDataset",
    "RecordImageTextDataset",
    "RecordTextDataset",
    "RecordPairDataset",
    "RecordTripletDataset",
    "GridImageCsvDataset",
    "GridImageDirDataset",
    "Hair10HairDataset",
    "CIFAR10Dataset",
    "MNISTDataset",
    "ImageMaskTransform",
    "ImageBoxesTransform",
    "DetectionCollate",
    "split_dataset",
    "train_val_split",
]
