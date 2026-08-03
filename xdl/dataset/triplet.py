"""Triplet datasets."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from .record import RecordDatasetBase
from .utils import PathLike, Record, Transform, apply_optional, load_image


class RecordTripletDataset(RecordDatasetBase):
    """Triplet dataset for metric learning or retrieval tasks."""

    def __init__(
        self,
        manifest_path: PathLike,
        anchor_image_key: str = "anchor_image",
        positive_image_key: str = "positive_image",
        negative_image_key: str = "negative_image",
        anchor_text_key: Optional[str] = None,
        positive_text_key: Optional[str] = None,
        negative_text_key: Optional[str] = None,
        base_dir: Optional[PathLike] = None,
        transform: Transform = None,
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
        self.anchor_image_key = anchor_image_key
        self.positive_image_key = positive_image_key
        self.negative_image_key = negative_image_key
        self.anchor_text_key = anchor_text_key
        self.positive_text_key = positive_text_key
        self.negative_text_key = negative_text_key
        self.transform = transform
        self.text_transform = text_transform
        self.image_mode = image_mode
        self.include_paths = bool(include_paths)

    def __getitem__(self, index: int) -> Record:
        base_index, record = self._record_at(index)
        anchor_path = self._resolve_record_path(record, self.anchor_image_key)
        positive_path = self._resolve_record_path(record, self.positive_image_key)
        negative_path = self._resolve_record_path(record, self.negative_image_key)
        sample: Record = {
            "anchor_image": apply_optional(
                self.transform,
                load_image(anchor_path, self.image_mode),
            ),
            "positive_image": apply_optional(
                self.transform,
                load_image(positive_path, self.image_mode),
            ),
            "negative_image": apply_optional(
                self.transform,
                load_image(negative_path, self.image_mode),
            ),
            "sample_id": self._sample_id_from_fallback(
                record,
                index=base_index,
                fallback=f"{anchor_path.stem}-{positive_path.stem}-{negative_path.stem}",
            ),
        }

        self._maybe_add_text(sample, record, "anchor_text", self.anchor_text_key)
        self._maybe_add_text(sample, record, "positive_text", self.positive_text_key)
        self._maybe_add_text(sample, record, "negative_text", self.negative_text_key)
        if self.include_paths:
            sample["anchor_image_path"] = str(anchor_path)
            sample["positive_image_path"] = str(positive_path)
            sample["negative_image_path"] = str(negative_path)
        return sample

    def _maybe_add_text(
        self,
        sample: Record,
        record: Mapping[str, Any],
        output_key: str,
        input_key: Optional[str],
    ) -> None:
        if input_key is None or record.get(input_key) in (None, ""):
            return
        value = str(record[input_key])
        sample[output_key] = self.text_transform(value) if self.text_transform is not None else value


__all__ = ["RecordTripletDataset"]
