"""Hairdata 数据集共享的数据增强变换和辅助方法。

三个 hairdata 数据集类的通用组件:
- hairdata.py (GridImageDataset, CSV 网格加载)
- hairdata3y.py (GridImageDataset, 目录列表加载)
- hairdata10hair.py (Hair10HairDataset, 三文件夹加载)
"""

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import torchvision.transforms as transforms


def create_pixel_transform(prob: float = 0.7) -> A.Compose:
    """Source/target 图像的增强变换 (三个数据集完全相同)。"""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=512),
            A.CenterCrop(512, 512),
            A.Affine(
                scale=(0.5, 1),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                p=0.8,
            ),
            A.OneOf(
                [
                    A.PixelDropout(dropout_prob=0.1, p=prob),
                    A.GaussNoise(mean_range=(0, 0), p=prob),
                    A.RandomShadow(shadow_roi=(0.1, 0.1, 0.9, 0.9), p=prob),
                ]
            ),
        ],
        additional_targets={"image0": "image"},
    )


def create_hair_transform() -> A.Compose:
    """Refer 图像的增强变换 (三个数据集完全相同)。"""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=512),
            A.CenterCrop(512, 512),
            A.Affine(scale=(0.9, 1.2), rotate=(-10, 10), p=0.7),
        ]
    )


class HairAugMixin:
    """Mixin 提供 imgaug / refer_imgaug 方法和共享的变换/归一化属性。

    使用方式:
        class MyDataset(HairAugMixin, Dataset):
            def __init__(self, ..., crop_size=768):
                self._init_hair_aug(crop_size=crop_size)
                # ... 数据集特定的初始化
    """

    def _init_hair_aug(self, crop_size: int | None = 768, prob: float = 0.7) -> None:
        """初始化共享的归一化、张量转换和增强变换。

        Args:
            crop_size: imgaug/refer_imgaug 中顶角裁剪尺寸 (None 表示不裁剪)。
            prob: 像素级增强概率。
        """
        self._hair_crop_size = crop_size
        self.norm = transforms.Normalize([0.5], [0.5])
        self.to_tensor = transforms.ToTensor()
        self.pixel_transform = create_pixel_transform(prob)
        self.hair_transform = create_hair_transform()

    def refer_imgaug(self, image):
        """对 refer 图像应用裁剪 + 缩放 + 颜色转换 + 增强 + 归一化。"""
        if self._hair_crop_size is not None:
            image = image[: self._hair_crop_size, : self._hair_crop_size]
        image = cv2.resize(image, [512, 512])
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hair_transform(image=image)
        return self.norm(self.to_tensor(results["image"] / 255.0))

    def imgaug(self, source_image, target_image):
        """对 source/target 图像对应用裁剪 + 缩放 + 颜色转换 + 增强 + 归一化。"""
        if self._hair_crop_size is not None:
            source_image = source_image[: self._hair_crop_size, : self._hair_crop_size]
            target_image = target_image[: self._hair_crop_size, : self._hair_crop_size]
        source_image = cv2.resize(source_image, [512, 512])
        target_image = cv2.resize(target_image, [512, 512])
        if len(source_image.shape) == 3 and source_image.shape[2] == 3:
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        if len(target_image.shape) == 3 and target_image.shape[2] == 3:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        results = self.pixel_transform(image=source_image, image0=target_image)
        return (
            self.norm(self.to_tensor(results["image"] / 255.0)),
            self.norm(self.to_tensor(results["image0"] / 255.0)),
        )
