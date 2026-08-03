"""Image editing datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

from .collate import ImageEditCollate
from .transforms import PairedImageTransform
from .utils import build_sample_id, first_present_value, load_manifest_context, resolve_path

IMAGE_KEYS = ("source_image", "target_image", "reference_image")
REQUIRED_IMAGE_KEYS = ("source_image", "target_image")
MASK_KEYS = ("edit_mask", "mask", "mask_image")
PROMPT_KEYS = ("prompt", "text", "caption")


class ImageEditDataset(Dataset[Dict[str, Any]]):
    """Dataset for image editing records with source/target/reference/mask fields."""

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
        (
            self.manifest_path,
            self.base_dir,
            self.records,
        ) = load_manifest_context(
            manifest_path,
            base_dir=base_dir,
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
        # 可选 reference 图缺失时补零, 让 batch schema 对下游模型保持固定.
        for key in self.image_keys:
            if key not in image_tensors:
                image_tensors[key] = torch.zeros_like(source_tensor)

        mask_tensor = next(iter(mask_tensors.values()), None)
        has_mask = mask_tensor is not None
        if mask_tensor is None:
            # 没有 edit mask 时返回全零 mask 和 has_mask=False, 避免下游反复判断字段是否存在.
            mask_tensor = torch.zeros(
                (1, source_tensor.shape[-2], source_tensor.shape[-1]),
                dtype=source_tensor.dtype,
            )

        sample: Dict[str, Any] = {
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
            images[key] = Image.open(resolve_path(value, self.base_dir)).convert("RGB")
        return images

    def _load_masks(self, record: Mapping[str, Any]) -> Dict[str, Image.Image]:
        for key in self.mask_keys:
            value = record.get(key)
            if value not in (None, ""):
                return {key: Image.open(resolve_path(value, self.base_dir)).convert("L")}
        return {}

    def _prompt(self, record: Mapping[str, Any]) -> str:
        value = first_present_value(record, self.prompt_keys, default="")
        return str(value) if value not in (None, "") else ""

    def _sample_id(self, record: Mapping[str, Any], index: int) -> str:
        source_value = record.get(self.image_keys[0])
        fallback = Path(str(source_value)).stem if source_value not in (None, "") else None
        return build_sample_id(
            record,
            index=index,
            sample_id_key=self.sample_id_key,
            fallback=fallback,
        )


ManifestImageEditDataset = ImageEditDataset
ManifestImageEditCollate = ImageEditCollate

__all__ = [
    "ImageEditDataset",
    "PairedImageTransform",
    "ImageEditCollate",
    "ManifestImageEditDataset",
    "ManifestImageEditCollate",
]
