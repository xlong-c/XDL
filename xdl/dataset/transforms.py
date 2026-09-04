"""Dataset transforms that must keep paired annotations in sync."""

from __future__ import annotations

import random
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageOps

from .utils import to_image_tensor, to_mask_tensor_float, to_mask_tensor_int

_to_image_tensor = to_image_tensor
_to_edit_mask_tensor = to_mask_tensor_float
_to_segmentation_mask_tensor = to_mask_tensor_int


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
        do_flip = self.random_flip and (torch.rand(1).item() < 0.5)
        image_tensors = {
            key: _to_image_tensor(
                self._prepare_image(image.convert("RGB"), do_flip=do_flip),
                normalize=self.normalize,
            )
            for key, image in images.items()
        }
        mask_tensors = {
            key: _to_edit_mask_tensor(self._prepare_image(mask.convert("L"), do_flip=do_flip))
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
        do_flip = self.random_flip and (torch.rand(1).item() < 0.5)
        # image 可以双线性插值, mask 必须用 nearest, 否则类别 id 会被插值污染.
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
        return _to_image_tensor(image, normalize=self.normalize), _to_segmentation_mask_tensor(mask)

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
        do_flip = self.random_flip and (torch.rand(1).item() < 0.5)
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


__all__ = ["PairedImageTransform", "ImageMaskTransform", "ImageBoxesTransform"]
