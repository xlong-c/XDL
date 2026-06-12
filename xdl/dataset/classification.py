"""Classification datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .record import RecordDatasetBase
from .utils import (
    PathLike,
    Transform,
    apply_optional,
    build_label_mapping,
    load_image,
    normalize_extensions,
    parse_sequence_field,
    require_record_keys,
    target_from_label,
)


class ImageFolderClassificationDataset(Dataset[Tuple[Any, Any]]):
    """Classification dataset for ``root/class_name/image`` directory layouts."""

    def __init__(
        self,
        root: PathLike,
        transform: Transform = None,
        target_transform: Transform = None,
        extensions: Optional[Sequence[str]] = None,
        class_to_idx: Optional[Mapping[str, int]] = None,
        image_mode: str = "RGB",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = normalize_extensions(extensions)
        self.image_mode = image_mode

        if class_to_idx is None:
            class_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
            self.class_to_idx = {path.name: idx for idx, path in enumerate(class_dirs)}
        else:
            self.class_to_idx = {str(key): int(value) for key, value in class_to_idx.items()}

        self.classes = [
            class_name
            for class_name, _idx in sorted(
                self.class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No image samples found under: {self.root}")

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        for class_name, target in sorted(self.class_to_idx.items(), key=lambda item: item[1]):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in self.extensions:
                    samples.append((image_path, target))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_path, target = self.samples[index]
        image = load_image(image_path, self.image_mode)
        image = apply_optional(self.transform, image)
        target = apply_optional(self.target_transform, target)
        return image, target


class RecordClassificationDataset(RecordDatasetBase):
    """Classification dataset backed by JSONL/JSON/CSV records."""

    def __init__(
        self,
        manifest_path: PathLike,
        image_key: str = "image",
        label_key: str = "label",
        base_dir: Optional[PathLike] = None,
        transform: Transform = None,
        target_transform: Transform = None,
        class_to_idx: Optional[Mapping[str, int]] = None,
        image_mode: str = "RGB",
        repeat: int = 1,
    ) -> None:
        super().__init__(manifest_path, base_dir=base_dir, repeat=repeat)
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        self.target_transform = target_transform
        self.image_mode = image_mode

        labels = [record[self.label_key] for record in self.records]
        self.class_to_idx = build_label_mapping(labels, class_to_idx)
        self.classes = (
            [
                label
                for label, _idx in sorted(
                    self.class_to_idx.items(),
                    key=lambda item: item[1],
                )
            ]
            if self.class_to_idx is not None
            else []
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        _base_index, record = self._record_at(index)
        require_record_keys(record, (self.image_key, self.label_key))

        image_path = self._resolve_record_path(record, self.image_key)
        image = load_image(image_path, self.image_mode)
        target = target_from_label(record[self.label_key], self.class_to_idx)
        image = apply_optional(self.transform, image)
        target = apply_optional(self.target_transform, target)
        return image, target


class RecordMultiLabelClassificationDataset(RecordDatasetBase):
    """Multi-label classification dataset backed by image + labels records."""

    def __init__(
        self,
        manifest_path: PathLike,
        image_key: str = "image",
        labels_key: str = "labels",
        base_dir: Optional[PathLike] = None,
        transform: Transform = None,
        target_transform: Transform = None,
        class_to_idx: Optional[Mapping[str, int]] = None,
        num_classes: Optional[int] = None,
        label_delimiter: str = ",",
        image_mode: str = "RGB",
        dtype: torch.dtype = torch.float32,
        repeat: int = 1,
    ) -> None:
        super().__init__(manifest_path, base_dir=base_dir, repeat=repeat)
        self.image_key = image_key
        self.labels_key = labels_key
        self.transform = transform
        self.target_transform = target_transform
        self.label_delimiter = label_delimiter
        self.image_mode = image_mode
        self.dtype = dtype

        label_lists = [
            self._coerce_labels(record.get(self.labels_key))
            for record in self.records
        ]
        # 多标签模板需要先扫描一次 manifest, 才能从字符串 label 推断 class_to_idx.
        flat_labels = [item for labels in label_lists for item in labels]
        self.class_to_idx = build_label_mapping(flat_labels, class_to_idx) if flat_labels else None
        if self.class_to_idx is not None:
            inferred_num_classes = len(self.class_to_idx)
            self.classes = [
                label
                for label, _idx in sorted(
                    self.class_to_idx.items(),
                    key=lambda item: item[1],
                )
            ]
        else:
            inferred_num_classes = (
                max((int(item) for item in flat_labels), default=-1) + 1
                if flat_labels
                else 0
            )
            self.classes = []
        self.num_classes = inferred_num_classes if num_classes is None else int(num_classes)
        if self.num_classes < inferred_num_classes:
            raise ValueError(
                f"num_classes={self.num_classes} is smaller than inferred class count "
                f"{inferred_num_classes}"
            )

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor]:
        _base_index, record = self._record_at(index)
        require_record_keys(record, (self.image_key, self.labels_key))

        image_path = self._resolve_record_path(record, self.image_key)
        image = load_image(image_path, self.image_mode)
        labels = self._coerce_labels(record[self.labels_key])
        target = torch.zeros(self.num_classes, dtype=self.dtype)
        for label in labels:
            class_index = target_from_label(label, self.class_to_idx)
            if class_index >= self.num_classes:
                raise ValueError(
                    f"Label index {class_index} exceeds num_classes={self.num_classes}"
                )
            target[class_index] = 1

        image = apply_optional(self.transform, image)
        target = apply_optional(self.target_transform, target)
        return image, target

    def _coerce_labels(self, value: Any) -> List[Any]:
        return parse_sequence_field(value, delimiter=self.label_delimiter)


__all__ = [
    "ImageFolderClassificationDataset",
    "RecordClassificationDataset",
    "RecordMultiLabelClassificationDataset",
]
