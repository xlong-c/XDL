# -*- coding: utf-8 -*-
import cv2
from torch.utils.data import Dataset
import os
import random
from xdl.utils import path_win2wsl

from .hair_transforms import HairAugMixin


class GridImageDataset(HairAugMixin, Dataset):
    """一个从目录列表加载图像的数据集类，固定为1行3列格式。"""

    def __init__(self, dir_list):
        self.dir_list = dir_list

        self.image_paths = []
        for dir_path in dir_list:
            if not os.path.exists(dir_path):
                print(f"警告: 目录不存在: {dir_path}")
                continue

            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self.image_paths.append(os.path.join(root, file))

        if not self.image_paths:
            raise ValueError(f"在目录列表 {dir_list} 中未找到任何图像文件")

        self.grid_cols = 3
        self.grid_rows = 1
        self.tile_width = 512
        self.tile_height = 512
        self.image_width = self.tile_width * self.grid_cols

        self._init_hair_aug(crop_size=512)

        print(f"Loaded {len(self.image_paths)} images from directories: {dir_list}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img_total = cv2.imread(image_path)

        if img_total is None:
            raise ValueError(f"无法加载图像: {image_path}")

        top = 0
        bottom = self.tile_height

        left_target = 0 * self.tile_width
        right_target = left_target + self.tile_width
        target_image = img_total[top:bottom, left_target:right_target]

        left_refer = 1 * self.tile_width
        right_refer = left_refer + self.tile_width
        refer_image = img_total[top:bottom, left_refer:right_refer]

        left_source = 2 * self.tile_width
        right_source = left_source + self.tile_width
        source_image = img_total[top:bottom, left_source:right_source]

        source_image, target_image = self.imgaug(source_image, target_image)
        refer_image = self.refer_imgaug(refer_image)

        batch = {
            "source_pixel_values": source_image,
            "target_pixel_values": target_image,
            "refer_pixel_values": refer_image,
        }
        return batch

    @staticmethod
    def get_image_paths_from_dirs(dir_list):
        image_paths = []
        for dir_path in dir_list:
            if not os.path.exists(dir_path):
                print(f"警告: 目录不存在: {dir_path}")
                continue

            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        image_paths.append(os.path.join(root, file))
        return image_paths


if __name__ == "__main__":
    dset = GridImageDataset(
        [
            path_win2wsl(r"F:\dataset\select_clein_lq_r"),
        ]
    )

    import random
    import matplotlib.pyplot as plt
    import numpy as np

    sample_idx = random.randint(0, len(dset) - 1)
    print(f"\n正在展示第 {sample_idx} 个样本（共 {len(dset)} 个样本）")

    sample = dset[0]

    print(f"Source image shape: {sample['source_pixel_values'].shape}")
    print(f"Target image shape: {sample['target_pixel_values'].shape}")
    print(f"Refer image shape: {sample['refer_pixel_values'].shape}")

    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    def denormalize(tensor):
        return (tensor * 0.5 + 0.5).clamp(0, 1)

    source_img = denormalize(sample["source_pixel_values"]).permute(1, 2, 0).numpy()
    target_img = denormalize(sample["target_pixel_values"]).permute(1, 2, 0).numpy()
    refer_img = denormalize(sample["refer_pixel_values"]).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(source_img)
    axes[0].set_title("Source Image (第3列)")
    axes[0].axis("off")

    axes[1].imshow(target_img)
    axes[1].set_title("Target Image (第1列)")
    axes[1].axis("off")

    axes[2].imshow(refer_img)
    axes[2].set_title("Refer Image (第2列)")
    axes[2].axis("off")

    plt.tight_layout()

    output_path = "dataset_sample_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n样本可视化已保存到: {output_path}")

    plt.close()

    print("\n样本展示完成！")
