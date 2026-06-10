"""General-purpose dataset templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset

from .manifest_image_edit import load_manifest_records

PathLike = Union[str, Path]
Record = Dict[str, Any]
Transform = Optional[Callable[[Any], Any]]

DEFAULT_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)
DEFAULT_TEXT_KEYS: Tuple[str, ...] = ("text", "prompt", "caption")


def _normalize_extensions(extensions: Optional[Sequence[str]]) -> Tuple[str, ...]:
    values = extensions or DEFAULT_IMAGE_EXTENSIONS
    return tuple(
        item.lower() if str(item).startswith(".") else f".{str(item).lower()}"
        for item in values
    )


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _load_image(path: Path, image_mode: str) -> Image.Image:
    return Image.open(path).convert(image_mode)


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


class ManifestRecordDataset(Dataset[Record]):
    """Load JSONL/JSON/CSV manifest records as dictionaries."""

    def __init__(
        self,
        manifest_path: PathLike,
        transform: Optional[Callable[[Record], Any]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.records = load_manifest_records(self.manifest_path)
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        record = dict(self.records[index])
        return self.transform(record) if self.transform is not None else record


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
        self.extensions = _normalize_extensions(extensions)
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
        image = _load_image(image_path, self.image_mode)
        image = _apply_optional(self.transform, image)
        target = _apply_optional(self.target_transform, target)
        return image, target


class ManifestClassificationDataset(Dataset[Tuple[Any, Any]]):
    """Classification dataset backed by a JSONL/JSON/CSV manifest."""

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
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.base_dir = (
            Path(base_dir).expanduser().resolve()
            if base_dir is not None
            else self.manifest_path.parent
        )
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        self.target_transform = target_transform
        self.image_mode = image_mode
        self.records = load_manifest_records(self.manifest_path)
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record = self.records[index]
        if self.image_key not in record:
            raise KeyError(f"Manifest record requires image key '{self.image_key}'")
        if self.label_key not in record:
            raise KeyError(f"Manifest record requires label key '{self.label_key}'")

        image_path = _resolve_path(record[self.image_key], self.base_dir)
        image = _load_image(image_path, self.image_mode)
        target = _target_from_label(record[self.label_key], self.class_to_idx)
        image = _apply_optional(self.transform, image)
        target = _apply_optional(self.target_transform, target)
        return image, target


class ManifestImageTextDataset(Dataset[Record]):
    """Image-text dataset backed by a JSONL/JSON/CSV manifest."""

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
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.base_dir = (
            Path(base_dir).expanduser().resolve()
            if base_dir is not None
            else self.manifest_path.parent
        )
        self.image_key = image_key
        self.text_keys = tuple(text_keys)
        self.sample_id_key = sample_id_key
        self.transform = transform
        self.text_transform = text_transform
        self.image_mode = image_mode
        self.include_image_path = bool(include_image_path)
        self.records = load_manifest_records(self.manifest_path)
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Record:
        record = self.records[index]
        if self.image_key not in record:
            raise KeyError(f"Manifest record requires image key '{self.image_key}'")

        image_path = _resolve_path(record[self.image_key], self.base_dir)
        image = _apply_optional(self.transform, _load_image(image_path, self.image_mode))
        text = self._select_text(record)
        text_value = (
            self.text_transform(text) if self.text_transform is not None else text
        )
        sample: Record = {
            "image": image,
            "text": text_value,
            "sample_id": self._sample_id(record, image_path, index),
        }
        if self.include_image_path:
            sample["image_path"] = str(image_path)
        return sample

    def _select_text(self, record: Mapping[str, Any]) -> str:
        for key in self.text_keys:
            value = record.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    def _sample_id(self, record: Mapping[str, Any], image_path: Path, index: int) -> str:
        if self.sample_id_key and record.get(self.sample_id_key) not in (None, ""):
            return str(record[self.sample_id_key])
        for key in ("sample_id", "id"):
            if record.get(key) not in (None, ""):
                return str(record[key])
        if image_path.name:
            return image_path.stem
        return str(index)
