"""Text record datasets."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from .record import RecordDatasetBase
from .utils import DEFAULT_TARGET_TEXT_KEYS, DEFAULT_TEXT_KEYS, PathLike, Record, first_present_value


class RecordTextDataset(RecordDatasetBase):
    """Text-only record dataset for language modeling or instruction tuning."""

    def __init__(
        self,
        manifest_path: PathLike,
        text_keys: Sequence[str] = DEFAULT_TEXT_KEYS,
        target_text_keys: Sequence[str] = DEFAULT_TARGET_TEXT_KEYS,
        sample_id_key: Optional[str] = None,
        base_dir: Optional[PathLike] = None,
        text_transform: Optional[Callable[[str], Any]] = None,
        target_text_transform: Optional[Callable[[str], Any]] = None,
        include_record: bool = False,
        repeat: int = 1,
    ) -> None:
        super().__init__(
            manifest_path,
            base_dir=base_dir,
            sample_id_key=sample_id_key,
            repeat=repeat,
        )
        self.text_keys = tuple(text_keys)
        self.target_text_keys = tuple(target_text_keys)
        self.text_transform = text_transform
        self.target_text_transform = target_text_transform
        self.include_record = bool(include_record)

    def __getitem__(self, index: int) -> Record:
        base_index, record = self._record_at(index)
        text = self._select_text(record, self.text_keys)
        sample: Record = {
            "text": self.text_transform(text) if self.text_transform is not None else text,
            "sample_id": self._sample_id_from_fallback(record, index=base_index),
        }

        target_text = self._select_optional_text(record, self.target_text_keys)
        if target_text is not None:
            sample["target_text"] = (
                self.target_text_transform(target_text)
                if self.target_text_transform is not None
                else target_text
            )
        if self.include_record:
            sample["record"] = dict(record)
        return sample

    def _select_text(self, record: Mapping[str, Any], keys: Sequence[str]) -> str:
        value = first_present_value(record, keys)
        if value in (None, ""):
            raise KeyError(f"Manifest record requires one of text keys {list(keys)}")
        return str(value)

    def _select_optional_text(
        self,
        record: Mapping[str, Any],
        keys: Sequence[str],
    ) -> Optional[str]:
        value = first_present_value(record, keys)
        if value in (None, ""):
            return None
        return str(value)


__all__ = ["RecordTextDataset"]
