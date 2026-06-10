"""Manifest-backed multi-image edit dataset."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


IMAGE_KEYS: Tuple[str, ...] = ("source_image", "target_image", "reference_image")
REQUIRED_IMAGE_KEYS: Tuple[str, ...] = ("source_image", "target_image")
MASK_KEYS: Tuple[str, ...] = ("edit_mask", "mask", "mask_image")
PROMPT_KEYS: Tuple[str, ...] = ("prompt", "text", "caption")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise TypeError(f"JSONL record must be a mapping: {path}")
        records.append(dict(item))
    return records


def _read_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise TypeError(f"JSON manifest must be a list or mapping: {path}")

    if not all(isinstance(item, dict) for item in records):
        raise TypeError(f"JSON manifest records must be mappings: {path}")
    return [dict(item) for item in records]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_manifest_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load records from a jsonl, json, or csv manifest."""

    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(manifest_path)
    if suffix == ".json":
        return _read_json(manifest_path)
    if suffix == ".csv":
        return _read_csv(manifest_path)
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _to_image_tensor(image: Image.Image, *, normalize: bool) -> torch.Tensor:
    array = np.array(image, dtype=np.float32, copy=True)
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def _to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.array(mask, dtype=np.float32, copy=True)
    if array.ndim == 3:
        array = array[:, :, 0]
    return torch.from_numpy(array[None, ...]).contiguous() / 255.0


class PairedImageTransform:
    """Apply the same resize/crop/flip decisions to images and masks."""

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

    def __call__(
        self,
        images: Mapping[str, Image.Image],
        masks: Optional[Mapping[str, Image.Image]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        do_flip = self.random_flip and random.random() < 0.5
        image_tensors = {
            key: _to_image_tensor(
                self._prepare_image(image.convert("RGB"), do_flip=do_flip),
                normalize=self.normalize,
            )
            for key, image in images.items()
        }
        mask_tensors = {
            key: _to_mask_tensor(self._prepare_image(mask.convert("L"), do_flip=do_flip))
            for key, mask in (masks or {}).items()
        }
        return image_tensors, mask_tensors

    def _prepare_image(self, image: Image.Image, *, do_flip: bool) -> Image.Image:
        if self.center_crop:
            image = ImageOps.fit(
                image,
                (self.width, self.height),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
        else:
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        if do_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


class ManifestImageEditDataset(Dataset[Dict[str, Any]]):
    """Dataset for image editing manifests with source/target/reference/mask fields."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        height: int = 512,
        width: int = 512,
        center_crop: bool = True,
        random_flip: bool = False,
        repeat: int = 1,
        base_dir: Optional[Union[str, Path]] = None,
        image_keys: Optional[Sequence[str]] = None,
        required_image_keys: Optional[Sequence[str]] = None,
        mask_keys: Optional[Sequence[str]] = None,
        prompt_keys: Optional[Sequence[str]] = None,
        sample_id_key: Optional[str] = None,
        transform: Optional[PairedImageTransform] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.records = load_manifest_records(self.manifest_path)
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        self.base_dir = (
            Path(base_dir).expanduser().resolve()
            if base_dir is not None
            else self.manifest_path.parent
        )
        self.image_keys = tuple(image_keys or IMAGE_KEYS)
        self.required_image_keys = tuple(required_image_keys or REQUIRED_IMAGE_KEYS)
        self.mask_keys = tuple(mask_keys or MASK_KEYS)
        self.prompt_keys = tuple(prompt_keys or PROMPT_KEYS)
        self.sample_id_key = sample_id_key
        self.repeat = max(1, int(repeat))
        self.transform = transform or PairedImageTransform(
            height=height,
            width=width,
            center_crop=center_crop,
            random_flip=random_flip,
            normalize=True,
        )

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index % len(self.records)]
        images = self._load_images(record)
        masks = self._load_masks(record)
        image_tensors, mask_tensors = self.transform(images, masks)

        source_tensor = image_tensors[self.image_keys[0]]
        for key in self.image_keys:
            if key not in image_tensors:
                image_tensors[key] = torch.zeros_like(source_tensor)

        mask_tensor = next(iter(mask_tensors.values()), None)
        has_mask = mask_tensor is not None
        if mask_tensor is None:
            mask_tensor = torch.zeros(
                (1, source_tensor.shape[-2], source_tensor.shape[-1]),
                dtype=source_tensor.dtype,
            )

        sample = {
            key: image_tensors[key]
            for key in self.image_keys
        }
        sample.update(
            {
                "edit_mask": mask_tensor.to(torch.float32),
                "has_mask": torch.tensor(has_mask, dtype=torch.bool),
                "sample_id": self._sample_id(record, index),
                "prompt": self._prompt(record),
            }
        )
        return sample

    def _load_images(self, record: Mapping[str, Any]) -> Dict[str, Image.Image]:
        images: Dict[str, Image.Image] = {}
        for key in self.image_keys:
            value = record.get(key)
            if value in (None, ""):
                if key in self.required_image_keys:
                    raise KeyError(f"Manifest record requires image key '{key}'")
                continue
            images[key] = Image.open(_resolve_path(value, self.base_dir)).convert("RGB")
        return images

    def _load_masks(self, record: Mapping[str, Any]) -> Dict[str, Image.Image]:
        for key in self.mask_keys:
            value = record.get(key)
            if value not in (None, ""):
                return {key: Image.open(_resolve_path(value, self.base_dir)).convert("L")}
        return {}

    def _prompt(self, record: Mapping[str, Any]) -> str:
        for key in self.prompt_keys:
            value = record.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    def _sample_id(self, record: Mapping[str, Any], index: int) -> str:
        if self.sample_id_key and record.get(self.sample_id_key) not in (None, ""):
            return str(record[self.sample_id_key])
        for key in ("sample_id", "id"):
            if record.get(key) not in (None, ""):
                return str(record[key])
        source_value = record.get(self.image_keys[0])
        if source_value not in (None, ""):
            return Path(str(source_value)).stem
        return str(index)


class ManifestImageEditCollate:
    """Collate manifest image edit samples into a dict batch."""

    def __init__(self, keys: Optional[Iterable[str]] = None) -> None:
        self.keys = list(keys) if keys is not None else None

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        keys = self.keys if self.keys is not None else list(batch[0].keys())
        collated: Dict[str, Any] = {}
        for key in keys:
            values = [item[key] for item in batch]
            if torch.is_tensor(values[0]):
                collated[key] = default_collate(values)
            else:
                collated[key] = list(values)
        return collated
