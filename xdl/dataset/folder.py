"""Image folder datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from torch.utils.data import Dataset

from .utils import (
    PathLike,
    Record,
    Transform,
    apply_optional,
    collect_image_paths,
    load_image,
    normalize_extensions,
    path_sample_id,
)


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
            "image": apply_optional(
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


__all__ = ["ImageFolderDataset"]
