"""Image-text datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .record import RecordDatasetBase
from .utils import (
    DEFAULT_TEXT_KEYS,
    PathLike,
    Record,
    Transform,
    apply_optional,
    collect_image_paths,
    collect_sidecar_samples,
    first_present_value,
    load_image,
    load_manifest_context,
    normalize_extension,
    normalize_extensions,
    path_sample_id,
    resolve_record_path,
    sidecar_path_for_image,
)


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
        image = apply_optional(self.transform, load_image(image_path, self.image_mode))
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
            "image": apply_optional(
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
            idx = int(torch.randint(0, len(lines), (1,)).item())
            return lines[idx]
        raise ValueError(
            "text_selection must be one of: 'full', 'first_line', 'random_line'"
        )
class ImagePromptDataset(Dataset[tuple[Any, str]]):
    """Manifest-backed image/prompt dataset for multimodal training.

    The dataset accepts JSONL, JSON, or CSV manifests and emits a tuple of
    ``(transformed_image, prompt)`` for training loops that use positional
    batches.  Prompt fields are checked in the configured order.
    """

    def __init__(
        self,
        manifest_path: PathLike,
        transform: Transform = None,
        text_keys: Sequence[str] = DEFAULT_TEXT_KEYS,
        base_dir: Optional[PathLike] = None,
        repeat: int = 1,
    ) -> None:
        self.manifest_path, self.base_dir, self.records = load_manifest_context(
            manifest_path,
            base_dir=base_dir,
        )
        self.transform = transform
        self.text_keys = tuple(text_keys)
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def __getitem__(self, index: int) -> tuple[Any, str]:
        base_index = index % len(self.records)
        record = self.records[base_index]
        image_path = resolve_record_path(record, "image", self.base_dir)
        prompt = first_present_value(record, self.text_keys, default=None)
        if prompt is None or not str(prompt).strip():
            raise KeyError(
                f"Manifest record requires one non-empty prompt field: {self.text_keys}"
            )
        image = apply_optional(self.transform, load_image(image_path, "RGB"))
        return image, str(prompt).strip()


__all__ = ["RecordImageTextDataset", "ImageTextSidecarDataset", "ImagePromptDataset"]
