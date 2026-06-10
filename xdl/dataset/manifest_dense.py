"""Manifest-backed dense prediction datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from ._manifest import load_manifest_records, resolve_path

PathLike = Union[str, Path]
Record = Dict[str, Any]
Transform = Optional[Callable[[Any], Any]]


def _to_image_tensor(image: Image.Image, *, normalize: bool) -> torch.Tensor:
    array = np.array(image, dtype=np.float32, copy=True)
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def _to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.array(mask, dtype=np.int64, copy=True)
    if array.ndim == 3:
        array = array[:, :, 0]
    return torch.from_numpy(array).contiguous()


def _coerce_boxes(value: Any) -> List[List[float]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Sequence):
        raise TypeError("boxes must be a sequence of [x1, y1, x2, y2] items")
    boxes: List[List[float]] = []
    for item in value:
        if not isinstance(item, Sequence) or len(item) != 4:
            raise TypeError("each box must be a sequence of length 4")
        boxes.append([float(coord) for coord in item])
    return boxes


def _coerce_labels(value: Any, expected_length: int) -> List[int]:
    if value in (None, ""):
        return [0] * expected_length
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("labels must be a sequence")
    labels = [int(item) for item in value]
    if len(labels) != expected_length:
        raise ValueError("labels length must match boxes length")
    return labels


class ImageMaskTransform:
    """Apply shared resize/crop/flip decisions to an image-mask pair."""

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        center_crop: bool = True,
        random_flip: bool = False,
        normalize: bool = True,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.center_crop = bool(center_crop)
        self.random_flip = bool(random_flip)
        self.normalize = bool(normalize)

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        do_flip = self.random_flip and random.random() < 0.5
        image = self._prepare_image(
            image,
            do_flip=do_flip,
            resample=Image.Resampling.BILINEAR,
        )
        mask = self._prepare_image(
            mask.convert("L"),
            do_flip=do_flip,
            resample=Image.Resampling.NEAREST,
        )
        return _to_image_tensor(image, normalize=self.normalize), _to_mask_tensor(mask)

    def _prepare_image(
        self,
        image: Image.Image,
        *,
        do_flip: bool,
        resample: Image.Resampling,
    ) -> Image.Image:
        if self.center_crop:
            image = ImageOps.fit(
                image,
                (self.width, self.height),
                method=resample,
                centering=(0.5, 0.5),
            )
        else:
            image = image.resize((self.width, self.height), resample)
        if do_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


class ImageBoxesTransform:
    """Apply shared resize/flip decisions to an image and detection boxes."""

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        random_flip: bool = False,
        normalize: bool = True,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.random_flip = bool(random_flip)
        self.normalize = bool(normalize)

    def __call__(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_width, original_height = image.size
        do_flip = self.random_flip and random.random() < 0.5
        resized = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        box_tensor = self._resize_boxes(
            boxes,
            original_width=original_width,
            original_height=original_height,
        )
        if do_flip:
            resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            box_tensor = self._flip_boxes(box_tensor, width=self.width)
        return _to_image_tensor(resized, normalize=self.normalize), box_tensor

    def _resize_boxes(
        self,
        boxes: Sequence[Sequence[float]],
        *,
        original_width: int,
        original_height: int,
    ) -> torch.Tensor:
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
        scale_x = self.width / float(original_width)
        scale_y = self.height / float(original_height)
        box_tensor[:, 0] *= scale_x
        box_tensor[:, 2] *= scale_x
        box_tensor[:, 1] *= scale_y
        box_tensor[:, 3] *= scale_y
        return box_tensor

    def _flip_boxes(self, boxes: torch.Tensor, *, width: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        return flipped


class ManifestSegmentationDataset(Dataset[Record]):
    """Segmentation dataset backed by an image-mask manifest."""

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
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.base_dir = (
            Path(base_dir).expanduser().resolve()
            if base_dir is not None
            else self.manifest_path.parent
        )
        self.image_key = image_key
        self.mask_key = mask_key
        self.transform = transform
        self.image_mode = image_mode
        self.include_paths = bool(include_paths)
        self.sample_id_key = sample_id_key
        self.records = load_manifest_records(self.manifest_path)
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Record:
        record = self.records[index]
        if self.image_key not in record:
            raise KeyError(f"Manifest record requires image key '{self.image_key}'")
        if self.mask_key not in record:
            raise KeyError(f"Manifest record requires mask key '{self.mask_key}'")

        image_path = resolve_path(record[self.image_key], self.base_dir)
        mask_path = resolve_path(record[self.mask_key], self.base_dir)
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
            "sample_id": self._sample_id(record, image_path, index),
        }
        if self.include_paths:
            sample["image_path"] = str(image_path)
            sample["mask_path"] = str(mask_path)
        return sample

    def _sample_id(self, record: Mapping[str, Any], image_path: Path, index: int) -> str:
        if self.sample_id_key and record.get(self.sample_id_key) not in (None, ""):
            return str(record[self.sample_id_key])
        for key in ("sample_id", "id"):
            if record.get(key) not in (None, ""):
                return str(record[key])
        if image_path.name:
            return image_path.stem
        return str(index)


class ManifestDetectionDataset(Dataset[Record]):
    """Detection dataset backed by an image + boxes manifest."""

    def __init__(
        self,
        manifest_path: PathLike,
        image_key: str = "image",
        boxes_key: str = "boxes",
        labels_key: str = "labels",
        base_dir: Optional[PathLike] = None,
        transform: Optional[ImageBoxesTransform] = None,
        image_mode: str = "RGB",
        include_paths: bool = False,
        sample_id_key: Optional[str] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.base_dir = (
            Path(base_dir).expanduser().resolve()
            if base_dir is not None
            else self.manifest_path.parent
        )
        self.image_key = image_key
        self.boxes_key = boxes_key
        self.labels_key = labels_key
        self.transform = transform
        self.image_mode = image_mode
        self.include_paths = bool(include_paths)
        self.sample_id_key = sample_id_key
        self.records = load_manifest_records(self.manifest_path)
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Record:
        record = self.records[index]
        if self.image_key not in record:
            raise KeyError(f"Manifest record requires image key '{self.image_key}'")
        if self.boxes_key not in record:
            raise KeyError(f"Manifest record requires boxes key '{self.boxes_key}'")

        image_path = resolve_path(record[self.image_key], self.base_dir)
        image = Image.open(image_path).convert(self.image_mode)
        boxes = _coerce_boxes(record[self.boxes_key])
        labels = _coerce_labels(record.get(self.labels_key), len(boxes))

        if self.transform is not None:
            image_value, box_tensor = self.transform(image, boxes)
        else:
            image_value = image
            box_tensor = torch.tensor(boxes, dtype=torch.float32)

        sample: Record = {
            "image": image_value,
            "boxes": box_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_id": self._sample_id(record, image_path, index),
        }
        if self.include_paths:
            sample["image_path"] = str(image_path)
        return sample

    def _sample_id(self, record: Mapping[str, Any], image_path: Path, index: int) -> str:
        if self.sample_id_key and record.get(self.sample_id_key) not in (None, ""):
            return str(record[self.sample_id_key])
        for key in ("sample_id", "id"):
            if record.get(key) not in (None, ""):
                return str(record[key])
        if image_path.name:
            return image_path.stem
        return str(index)


class DetectionCollate:
    """Collate detection samples while preserving variable-sized targets."""

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        collated: Dict[str, Any] = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if key in {"boxes", "labels"}:
                collated[key] = values
            elif torch.is_tensor(values[0]):
                collated[key] = default_collate(values)
            else:
                collated[key] = list(values)
        return collated
