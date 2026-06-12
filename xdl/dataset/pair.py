"""Pair datasets."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .record import RecordDatasetBase
from .utils import PathLike, Record, Transform, apply_optional, load_image


class RecordPairDataset(RecordDatasetBase):
    """Pair dataset for contrastive, siamese, or retrieval-style tasks."""

    def __init__(
        self,
        manifest_path: PathLike,
        image_a_key: str = "image_a",
        image_b_key: str = "image_b",
        label_key: Optional[str] = "label",
        text_a_key: Optional[str] = None,
        text_b_key: Optional[str] = None,
        base_dir: Optional[PathLike] = None,
        transform: Transform = None,
        target_transform: Transform = None,
        text_transform: Optional[Callable[[str], Any]] = None,
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
        self.image_a_key = image_a_key
        self.image_b_key = image_b_key
        self.label_key = label_key
        self.text_a_key = text_a_key
        self.text_b_key = text_b_key
        self.transform = transform
        self.target_transform = target_transform
        self.text_transform = text_transform
        self.image_mode = image_mode
        self.include_paths = bool(include_paths)

    def __getitem__(self, index: int) -> Record:
        base_index, record = self._record_at(index)
        image_a_path = self._resolve_record_path(record, self.image_a_key)
        image_b_path = self._resolve_record_path(record, self.image_b_key)
        sample: Record = {
            "image_a": apply_optional(
                self.transform,
                load_image(image_a_path, self.image_mode),
            ),
            "image_b": apply_optional(
                self.transform,
                load_image(image_b_path, self.image_mode),
            ),
        }
        if self.label_key is not None and self.label_key in record:
            target = record[self.label_key]
            sample["label"] = apply_optional(self.target_transform, target)
        if self.text_a_key is not None and record.get(self.text_a_key) not in (None, ""):
            sample["text_a"] = self._transform_text(record[self.text_a_key])
        if self.text_b_key is not None and record.get(self.text_b_key) not in (None, ""):
            sample["text_b"] = self._transform_text(record[self.text_b_key])
        sample["sample_id"] = self._sample_id_from_fallback(
            record,
            index=base_index,
            fallback=f"{image_a_path.stem}-{image_b_path.stem}",
        )
        if self.include_paths:
            sample["image_a_path"] = str(image_a_path)
            sample["image_b_path"] = str(image_b_path)
        return sample

    def _transform_text(self, value: Any) -> Any:
        text = str(value)
        if self.text_transform is not None:
            return self.text_transform(text)
        return text


__all__ = ["RecordPairDataset"]
