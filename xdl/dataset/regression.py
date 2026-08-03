"""Regression datasets."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .record import RecordDatasetBase
from .utils import PathLike, Transform, apply_optional, load_image, require_record_keys


class RecordRegressionDataset(RecordDatasetBase):
    """Regression dataset backed by JSONL/JSON/CSV records."""

    def __init__(
        self,
        manifest_path: PathLike,
        image_key: str = "image",
        target_key: str = "target",
        base_dir: Optional[PathLike] = None,
        transform: Transform = None,
        target_transform: Transform = None,
        image_mode: str = "RGB",
        dtype: Callable[[Any], Any] = float,
        repeat: int = 1,
    ) -> None:
        super().__init__(manifest_path, base_dir=base_dir, repeat=repeat)
        self.image_key = image_key
        self.target_key = target_key
        self.transform = transform
        self.target_transform = target_transform
        self.image_mode = image_mode
        self.dtype = dtype

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        _base_index, record = self._record_at(index)
        require_record_keys(record, (self.image_key, self.target_key))

        image_path = self._resolve_record_path(record, self.image_key)
        image = load_image(image_path, self.image_mode)
        target = self.dtype(record[self.target_key])
        image = apply_optional(self.transform, image)
        target = apply_optional(self.target_transform, target)
        return image, target


__all__ = ["RecordRegressionDataset"]
