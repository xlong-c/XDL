"""Record-backed dataset base classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from torch.utils.data import Dataset

from .utils import PathLike, Record, build_sample_id, load_manifest_context, resolve_record_path


class RecordDatasetBase(Dataset[Any]):
    """Shared record dataset utilities."""

    def __init__(
        self,
        manifest_path: PathLike,
        *,
        base_dir: Optional[PathLike] = None,
        sample_id_key: Optional[str] = None,
        repeat: int = 1,
    ) -> None:
        (
            self.manifest_path,
            self.base_dir,
            self.records,
        ) = load_manifest_context(manifest_path, base_dir=base_dir)
        self.sample_id_key = sample_id_key
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def _record_at(self, index: int) -> tuple[int, Record]:
        # repeat 只扩展逻辑长度, 实际 record 仍回到原始 manifest 下标.
        base_index = index % len(self.records)
        return base_index, self.records[base_index]

    def _sample_id_from_path(self, record: Mapping[str, Any], path: Path, index: int) -> str:
        fallback = path.stem if path.name else None
        return build_sample_id(
            record,
            index=index,
            sample_id_key=self.sample_id_key,
            fallback=fallback,
        )

    def _sample_id_from_fallback(
        self,
        record: Mapping[str, Any],
        *,
        index: int,
        fallback: Optional[str] = None,
    ) -> str:
        return build_sample_id(
            record,
            index=index,
            sample_id_key=self.sample_id_key,
            fallback=fallback,
        )

    def _resolve_record_path(self, record: Mapping[str, Any], key: str) -> Path:
        return resolve_record_path(record, key, self.base_dir)


class RecordDataset(RecordDatasetBase):
    """Load JSONL/JSON/CSV records as dictionaries."""

    def __init__(
        self,
        manifest_path: PathLike,
        transform: Optional[Callable[[Record], Any]] = None,
        *,
        base_dir: Optional[PathLike] = None,
        repeat: int = 1,
    ) -> None:
        super().__init__(manifest_path, base_dir=base_dir, repeat=repeat)
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        _base_index, record = self._record_at(index)
        payload = dict(record)
        return self.transform(payload) if self.transform is not None else payload


__all__ = ["RecordDatasetBase", "RecordDataset"]
