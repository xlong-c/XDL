"""
数据集注册入口。

注意：部分数据集依赖可选三方库，导入失败时不会阻断整个框架。
"""

from xdl.utils.registry import register_dataset

_import_errors = {}
GridImageCsvDataset = None
GridImageDirDataset = None
Hair10HairDataset = None

from .basic import SyntheticClassificationDataset

register_dataset("SyntheticClassificationDataset")(SyntheticClassificationDataset)

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
    "GridImageCsvDataset",
    "GridImageDirDataset",
    "Hair10HairDataset",
]
