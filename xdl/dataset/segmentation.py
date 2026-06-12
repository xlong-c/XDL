"""Segmentation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .record import RecordDatasetBase
from .transforms import ImageMaskTransform
from .utils import (
    PathLike,
    Record,
    collect_image_paths,
    collect_sidecar_samples,
    load_image,
    normalize_extensions,
    path_sample_id,
    require_record_keys,
)


class ImageMaskSidecarDataset(Dataset[Record]):
    """Segmentation dataset for basename-aligned image and mask directories."""

    def __init__(
        self,
        root: Optional[PathLike] = None,
        image_root: Optional[PathLike] = None,
        mask_root: Optional[PathLike] = None,
        mask_extension: Optional[str] = None,
        transform: Optional[
            Callable[[Image.Image, Image.Image], Tuple[Any, Any]]
        ] = None,
        image_mode: str = "RGB",
        mask_mode: str = "L",
        extensions: Optional[Sequence[str]] = None,
        recursive: bool = True,
        include_paths: bool = True,
        missing_mask: str = "error",
        sample_id_from: str = "stem",
        repeat: int = 1,
    ) -> None:
        if root is None and image_root is None:
            raise ValueError("Either root or image_root must be provided")
        if root is not None and image_root is not None:
            raise ValueError("Use either root or image_root, not both")

        resolved_image_root = image_root if image_root is not None else root
        if resolved_image_root is None:
            raise ValueError("Either root or image_root must be provided")

        self.image_root = Path(resolved_image_root).expanduser().resolve()
        self.mask_root = (
            Path(mask_root).expanduser().resolve()
            if mask_root is not None
            else None
        )
        self.mask_extension = mask_extension
        self.transform = transform
        self.image_mode = image_mode
        self.mask_mode = mask_mode
        self.extensions = normalize_extensions(extensions)
        self.recursive = bool(recursive)
        self.include_paths = bool(include_paths)
        self.missing_mask = missing_mask
        self.sample_id_from = sample_id_from
        self.repeat = max(1, int(repeat))

        image_paths = collect_image_paths(
            self.image_root,
            extensions=self.extensions,
            recursive=self.recursive,
        )
        # sidecar mask 与 image 使用同一相对路径, 支持 images/a.png -> masks/a.png.
        self.samples = collect_sidecar_samples(
            image_paths,
            image_root=self.image_root,
            sidecar_root=self.mask_root,
            sidecar_extension=self.mask_extension,
            missing=self.missing_mask,
            sidecar_name="mask",
        )
        if not self.samples:
            raise ValueError(f"No image/mask sidecar samples found under: {self.image_root}")

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> Record:
        base_index = index % len(self.samples)
        image_path, mask_path = self.samples[base_index]
        image = load_image(image_path, self.image_mode)
        mask = Image.open(mask_path).convert(self.mask_mode)
        if self.transform is not None:
            image_value, mask_value = self.transform(image, mask)
        else:
            image_value = image
            mask_value = mask

        sample: Record = {
            "image": image_value,
            "mask": mask_value,
            "sample_id": path_sample_id(
                image_path,
                root=self.image_root,
                index=base_index,
                sample_id_from=self.sample_id_from,
            ),
        }
        if self.include_paths:
            sample["image_path"] = str(image_path)
            sample["mask_path"] = str(mask_path)
        return sample


class RecordSegmentationDataset(RecordDatasetBase):
    """Segmentation dataset backed by image-mask records."""

    def __init__(
        self,
        manifest_path: PathLike,
        image_key: str = "image",
        mask_key: str = "mask",
        base_dir: Optional[PathLike] = None,
        transform: Optional[ImageMaskTransform] = None,
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
        self.mask_key = mask_key
        self.transform = transform
        self.image_mode = image_mode
        self.include_paths = bool(include_paths)

    def __getitem__(self, index: int) -> Record:
        base_index, record = self._record_at(index)
        require_record_keys(record, (self.image_key, self.mask_key))

        image_path = self._resolve_record_path(record, self.image_key)
        mask_path = self._resolve_record_path(record, self.mask_key)
        image = Image.open(image_path).convert(self.image_mode)
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            image_value, mask_value = self.transform(image, mask)
        else:
            image_value = image
            mask_value = mask

        sample: Record = {
            "image": image_value,
            "mask": mask_value,
            "sample_id": self._sample_id_from_path(record, image_path, base_index),
        }
        if self.include_paths:
            sample["image_path"] = str(image_path)
            sample["mask_path"] = str(mask_path)
        return sample


__all__ = ["ImageMaskSidecarDataset", "RecordSegmentationDataset"]
