# -*- coding: utf-8 -*-
import os
import glob

import cv2
from torch.utils.data import Dataset

from .hair_transforms import HairAugMixin


class Hair10HairDataset(HairAugMixin, Dataset):
    """从三个文件夹加载图像的数据集类。"""

    def __init__(self, bimg_fold, zimg_fold, rimg_fold):
        self.bimg_fold = bimg_fold
        self.zimg_fold = zimg_fold
        self.rimg_fold = rimg_fold

        self.samples = []
        self._collect_samples()

        self._init_hair_aug(crop_size=None)

        print(f"Loaded {len(self.samples)} samples from {bimg_fold}")

    def _collect_samples(self):
        bimg_files = glob.glob(os.path.join(self.bimg_fold, "*.png"))

        for bimg_path in bimg_files:
            filename = os.path.basename(bimg_path)
            basename = os.path.splitext(filename)[0]

            if "_" not in basename:
                print(f"Warning: Skipping invalid filename {filename} (no underscore)")
                continue

            parts = basename.rsplit("_", 1)
            if len(parts) != 2:
                print(f"Warning: Skipping invalid filename {filename}")
                continue

            cls_id, index = parts

            target_path = os.path.join(self.zimg_fold, filename)
            refer_path = os.path.join(self.rimg_fold, f"{cls_id}.png")

            if not os.path.exists(target_path):
                print(
                    f"Warning: Target file not found for {filename}, expected at {target_path}"
                )
                continue

            if not os.path.exists(refer_path):
                print(
                    f"Warning: Refer file not found for {filename}, expected at {refer_path}"
                )
                continue

            self.samples.append(
                {
                    "cls_id": cls_id,
                    "index": index,
                    "source_path": bimg_path,
                    "target_path": target_path,
                    "refer_path": refer_path,
                }
            )

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid samples found in {self.bimg_fold}. "
                f"Please check the folder structure and file naming convention."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        source_image = cv2.imread(sample_info["source_path"])
        target_image = cv2.imread(sample_info["target_path"])
        refer_image = cv2.imread(sample_info["refer_path"])

        if source_image is None:
            raise ValueError(f"无法加载source图像: {sample_info['source_path']}")
        if target_image is None:
            raise ValueError(f"无法加载target图像: {sample_info['target_path']}")
        if refer_image is None:
            raise ValueError(f"无法加载refer图像: {sample_info['refer_path']}")

        source_image, target_image = self.imgaug(source_image, target_image)
        refer_image = self.refer_imgaug(refer_image)

        batch = {
            "source_pixel_values": source_image,
            "target_pixel_values": target_image,
            "refer_pixel_values": refer_image,
        }
        return batch


if __name__ == "__main__":
    bimg_fold = r"F:\dataset\10hair\bimg"
    zimg_fold = r"F:\dataset\10hair\zimg"
    rimg_fold = r"F:\dataset\10hair\rimg"

    try:
        dset = Hair10HairDataset(bimg_fold, zimg_fold, rimg_fold)

        import random

        import matplotlib.pyplot as plt

        sample_idx = random.randint(0, len(dset) - 1)
        print(f"\n正在展示第 {sample_idx} 个样本(共 {len(dset)} 个样本)")

        sample = dset[sample_idx]

        print(f"Source image shape: {sample['source_pixel_values'].shape}")
        print(f"Target image shape: {sample['target_pixel_values'].shape}")
        print(f"Refer image shape: {sample['refer_pixel_values'].shape}")

        plt.rcParams["font.sans-serif"] = [
            "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        def denormalize(tensor):
            return (tensor * 0.5 + 0.5).clamp(0, 1)

        source_img = (
            denormalize(sample["source_pixel_values"]).permute(1, 2, 0).numpy()
        )
        target_img = (
            denormalize(sample["target_pixel_values"]).permute(1, 2, 0).numpy()
        )
        refer_img = denormalize(sample["refer_pixel_values"]).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(source_img)
        axes[0].set_title("Source Image (bimg)")
        axes[0].axis("off")

        axes[1].imshow(target_img)
        axes[1].set_title("Target Image (zimg)")
        axes[1].axis("off")

        axes[2].imshow(refer_img)
        axes[2].set_title("Refer Image (rimg)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

        print("\n样本展示完成！")

    except Exception as e:
        print(f"错误: {e}")
        print("\n请检查：")
        print("1. 文件夹路径是否正确")
        print("2. bimg文件夹中是否存在 {cls_id}_{index}.png 格式的文件")
        print("3. zimg文件夹中是否存在对应的target文件")
        print("4. rimg文件夹中是否存在对应的 {cls_id}.png 文件")
