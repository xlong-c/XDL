"""General-purpose dataset templates."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from ._records import (
    build_sample_id,
    first_present_value,
    load_manifest_context,
    require_record_keys,
    resolve_record_path,
)
from ._paths import (
    collect_image_paths,
    collect_sidecar_samples,
    load_image,
    normalize_extension,
    normalize_extensions,
    path_sample_id,
    sidecar_path_for_image,
)

PathLike = Union[str, Path]
Record = Dict[str, Any]
Transform = Optional[Callable[[Any], Any]]

DEFAULT_TEXT_KEYS: Tuple[str, ...] = ("text", "prompt", "caption")
DEFAULT_TARGET_TEXT_KEYS: Tuple[str, ...] = (
    "target_text",
    "response",
    "completion",
    "answer",
)


def _apply_optional(transform: Transform, value: Any) -> Any:
    return transform(value) if transform is not None else value


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return str(value).strip() == str(int(value))


def _build_label_mapping(
    labels: Sequence[Any],
    class_to_idx: Optional[Mapping[str, int]],
) -> Optional[Dict[str, int]]:
    if class_to_idx is not None:
        return {str(key): int(value) for key, value in class_to_idx.items()}
    if all(_is_int_like(label) for label in labels):
        return None
    return {label: idx for idx, label in enumerate(sorted({str(item) for item in labels}))}


def _target_from_label(label: Any, label_mapping: Optional[Mapping[str, int]]) -> int:
    if label_mapping is None:
        return int(label)
    key = str(label)
    if key not in label_mapping:
        raise KeyError(f"Unknown label '{label}'")
    return int(label_mapping[key])


def _parse_sequence_field(
    value: Any,
    *,
    delimiter: str = ",",
) -> List[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") or stripped.startswith("("):
            decoded = json.loads(stripped)
            if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
                raise TypeError("Decoded sequence field must be a sequence")
            return list(decoded)
        return [item.strip() for item in stripped.split(delimiter) if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise TypeError("Expected a sequence-like field")


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
        image = _apply_optional(self.transform, image)
        target = _apply_optional(self.target_transform, target)
        return image, target


class ImageFolderDataset(Dataset[Record]):
    """Image-only dataset for plain image directories."""

    def __init__(
        self,
        root: PathLike,
        transform: Transform = None,
        extensions: Optional[Sequence[str]] = None,
        image_mode: str = "RGB",
        recursive: bool = True,
        include_paths: bool = True,
        sample_id_from: str = "stem",
        repeat: int = 1,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.extensions = normalize_extensions(extensions)
        self.image_mode = image_mode
        self.recursive = bool(recursive)
        self.include_paths = bool(include_paths)
        self.sample_id_from = sample_id_from
        self.repeat = max(1, int(repeat))
        self.image_paths = collect_image_paths(
            self.root,
            extensions=self.extensions,
            recursive=self.recursive,
        )
        if not self.image_paths:
            raise ValueError(f"No image samples found under: {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths) * self.repeat

    def __getitem__(self, index: int) -> Record:
        base_index = index % len(self.image_paths)
        image_path = self.image_paths[base_index]
        sample: Record = {
            "image": _apply_optional(
                self.transform,
                load_image(image_path, self.image_mode),
            ),
            "sample_id": path_sample_id(
                image_path,
                root=self.root,
                index=base_index,
                sample_id_from=self.sample_id_from,
            ),
        }
        if self.include_paths:
            sample["image_path"] = str(image_path)
        return sample


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
        self.class_to_idx = _build_label_mapping(labels, class_to_idx)
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
        target = _target_from_label(record[self.label_key], self.class_to_idx)
        image = _apply_optional(self.transform, image)
        target = _apply_optional(self.target_transform, target)
        return image, target


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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        _base_index, record = self._record_at(index)
        require_record_keys(record, (self.image_key, self.target_key))

        image_path = self._resolve_record_path(record, self.image_key)
        image = load_image(image_path, self.image_mode)
        target = self.dtype(record[self.target_key])
        image = _apply_optional(self.transform, image)
        target = _apply_optional(self.target_transform, target)
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
        self.class_to_idx = _build_label_mapping(flat_labels, class_to_idx) if flat_labels else None
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
            class_index = _target_from_label(label, self.class_to_idx)
            if class_index >= self.num_classes:
                raise ValueError(
                    f"Label index {class_index} exceeds num_classes={self.num_classes}"
                )
            target[class_index] = 1

        image = _apply_optional(self.transform, image)
        target = _apply_optional(self.target_transform, target)
        return image, target

    def _coerce_labels(self, value: Any) -> List[Any]:
        return _parse_sequence_field(value, delimiter=self.label_delimiter)


class RecordImageTextDataset(RecordDatasetBase):
    """Image-text dataset backed by JSONL/JSON/CSV records."""

    def __init__(
        self,
        manifest_path: PathLike,
        image_key: str = "image",
        text_keys: Sequence[str] = DEFAULT_TEXT_KEYS,
        sample_id_key: Optional[str] = None,
        base_dir: Optional[PathLike] = None,
        transform: Transform = None,
        text_transform: Optional[Callable[[str], Any]] = None,
        image_mode: str = "RGB",
        include_image_path: bool = True,
        repeat: int = 1,
    ) -> None:
        super().__init__(
            manifest_path,
            base_dir=base_dir,
            sample_id_key=sample_id_key,
            repeat=repeat,
        )
        self.image_key = image_key
        self.text_keys = tuple(text_keys)
        self.transform = transform
        self.text_transform = text_transform
        self.image_mode = image_mode
        self.include_image_path = bool(include_image_path)

    def __getitem__(self, index: int) -> Record:
        base_index, record = self._record_at(index)
        image_path = self._resolve_record_path(record, self.image_key)
        image = _apply_optional(self.transform, load_image(image_path, self.image_mode))
        text = self._select_text(record)
        text_value = self.text_transform(text) if self.text_transform is not None else text
        sample: Record = {
            "image": image,
            "text": text_value,
            "sample_id": self._sample_id_from_path(record, image_path, base_index),
        }
        if self.include_image_path:
            sample["image_path"] = str(image_path)
        return sample

    def _select_text(self, record: Mapping[str, Any]) -> str:
        value = first_present_value(record, self.text_keys, default="")
        return str(value) if value not in (None, "") else ""


class ImageTextSidecarDataset(Dataset[Record]):
    """Image-text dataset for basename-aligned image and text sidecar files."""

    def __init__(
        self,
        root: Optional[PathLike] = None,
        image_root: Optional[PathLike] = None,
        text_root: Optional[PathLike] = None,
        text_extension: str = ".txt",
        transform: Transform = None,
        text_transform: Optional[Callable[[str], Any]] = None,
        extensions: Optional[Sequence[str]] = None,
        image_mode: str = "RGB",
        recursive: bool = True,
        include_paths: bool = True,
        missing_text: str = "error",
        text_selection: str = "full",
        sample_id_from: str = "stem",
        repeat: int = 1,
    ) -> None:
        if root is None and image_root is None:
            raise ValueError("Either root or image_root must be provided")
        if root is not None and image_root is not None:
            raise ValueError("Use either root or image_root, not both")
        if missing_text not in {"error", "skip"}:
            raise ValueError("missing_text must be one of: 'error', 'skip'")
        if text_selection not in {"full", "first_line", "random_line"}:
            raise ValueError(
                "text_selection must be one of: 'full', 'first_line', 'random_line'"
            )

        resolved_image_root = image_root if image_root is not None else root
        if resolved_image_root is None:
            raise ValueError("Either root or image_root must be provided")

        self.image_root = Path(resolved_image_root).expanduser().resolve()
        self.text_root = (
            Path(text_root).expanduser().resolve()
            if text_root is not None
            else None
        )
        self.text_extension = normalize_extension(text_extension)
        self.transform = transform
        self.text_transform = text_transform
        self.extensions = normalize_extensions(extensions)
        self.image_mode = image_mode
        self.recursive = bool(recursive)
        self.include_paths = bool(include_paths)
        self.missing_text = missing_text
        self.text_selection = text_selection
        self.sample_id_from = sample_id_from
        self.repeat = max(1, int(repeat))

        image_paths = collect_image_paths(
            self.image_root,
            extensions=self.extensions,
            recursive=self.recursive,
        )
        # image_root/text_root 可以分目录, collect_sidecar_samples 会按相对路径保持层级对齐.
        self.samples = collect_sidecar_samples(
            image_paths,
            image_root=self.image_root,
            sidecar_root=self.text_root,
            sidecar_extension=self.text_extension,
            missing=self.missing_text,
            sidecar_name="text",
        )
        if not self.samples:
            raise ValueError(f"No image/text sidecar samples found under: {self.image_root}")

    def _text_path_for_image(self, image_path: Path) -> Path:
        return sidecar_path_for_image(
            image_path,
            image_root=self.image_root,
            sidecar_root=self.text_root,
            sidecar_extension=self.text_extension,
        )

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> Record:
        base_index = index % len(self.samples)
        image_path, text_path = self.samples[base_index]
        text = self._read_text(text_path)
        sample: Record = {
            "image": _apply_optional(
                self.transform,
                load_image(image_path, self.image_mode),
            ),
            "text": self.text_transform(text) if self.text_transform is not None else text,
            "sample_id": path_sample_id(
                image_path,
                root=self.image_root,
                index=base_index,
                sample_id_from=self.sample_id_from,
            ),
        }
        if self.include_paths:
            sample["image_path"] = str(image_path)
            sample["text_path"] = str(text_path)
        return sample

    def _read_text(self, text_path: Path) -> str:
        content = text_path.read_text(encoding="utf-8")
        if self.text_selection == "full":
            return content.strip()

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return ""
        if self.text_selection == "first_line":
            return lines[0]
        if self.text_selection == "random_line":
            return random.choice(lines)
        raise ValueError(
            "text_selection must be one of: 'full', 'first_line', 'random_line'"
        )


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
            "image_a": _apply_optional(
                self.transform,
                load_image(image_a_path, self.image_mode),
            ),
            "image_b": _apply_optional(
                self.transform,
                load_image(image_b_path, self.image_mode),
            ),
        }
        if self.label_key is not None and self.label_key in record:
            target = record[self.label_key]
            sample["label"] = _apply_optional(self.target_transform, target)
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
            "anchor_image": _apply_optional(
                self.transform,
                load_image(anchor_path, self.image_mode),
            ),
            "positive_image": _apply_optional(
                self.transform,
                load_image(positive_path, self.image_mode),
            ),
            "negative_image": _apply_optional(
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


ManifestDatasetBase = RecordDatasetBase
ManifestRecordDataset = RecordDataset
ManifestClassificationDataset = RecordClassificationDataset
ManifestRegressionDataset = RecordRegressionDataset
ManifestMultiLabelClassificationDataset = RecordMultiLabelClassificationDataset
ManifestImageTextDataset = RecordImageTextDataset
ManifestTextDataset = RecordTextDataset
ManifestPairDataset = RecordPairDataset
ManifestTripletDataset = RecordTripletDataset
