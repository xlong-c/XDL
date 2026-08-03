"""Detection datasets."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

import torch
from PIL import Image

from .record import RecordDatasetBase
from .transforms import ImageBoxesTransform
from .utils import PathLike, Record, require_record_keys


def _coerce_boxes(value: Any) -> List[List[float]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Sequence):
        raise TypeError("boxes must be a sequence of [x1, y1, x2, y2] items")
    boxes: List[List[float]] = []
    for item in value:
        if not isinstance(item, Sequence) or len(item) != 4:
            raise TypeError("each box must be a sequence of length 4")
        boxes.append([float(coord) for coord in item])
    return boxes


def _coerce_labels(value: Any, expected_length: int) -> List[int]:
    if value in (None, ""):
        return [0] * expected_length
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("labels must be a sequence")
    labels = [int(item) for item in value]
    if len(labels) != expected_length:
        raise ValueError("labels length must match boxes length")
    return labels


class RecordDetectionDataset(RecordDatasetBase):
    """Detection dataset backed by image + boxes records."""

    def __init__(
        self,
        manifest_path: PathLike,
        image_key: str = "image",
        boxes_key: str = "boxes",
        labels_key: str = "labels",
        base_dir: Optional[PathLike] = None,
        transform: Optional[ImageBoxesTransform] = None,
        image_mode: str = "RGB",
        include_paths: bool = False,
        sample_id_key: Optional[str] = None,
        repeat: int = 1,
    ) -> None:
        super().__init__(
            manifest_path,
            base_dir=base_dir,
            sample_id_key=sample_id_key,
            repeat=repeat,
        )
        self.image_key = image_key
        self.boxes_key = boxes_key
        self.labels_key = labels_key
        self.transform = transform
        self.image_mode = image_mode
        self.include_paths = bool(include_paths)

    def __getitem__(self, index: int) -> Record:
        base_index, record = self._record_at(index)
        require_record_keys(record, (self.image_key, self.boxes_key))

        image_path = self._resolve_record_path(record, self.image_key)
        image = Image.open(image_path).convert(self.image_mode)
        boxes = _coerce_boxes(record[self.boxes_key])
        labels = _coerce_labels(record.get(self.labels_key), len(boxes))

        if self.transform is not None:
            image_value, box_tensor = self.transform(image, boxes)
        else:
            image_value = image
            box_tensor = torch.tensor(boxes, dtype=torch.float32)

        sample: Record = {
            "image": image_value,
            "boxes": box_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_id": self._sample_id_from_path(record, image_path, base_index),
        }
        if self.include_paths:
            sample["image_path"] = str(image_path)
        return sample


__all__ = ["RecordDetectionDataset"]
