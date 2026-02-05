# -*- coding: utf-8 -*-
import cv2
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import csv
import random
from xdl.utils import path_win2wsl


class GridImageDataset(Dataset):
    """
    一个从目录列表加载图像的数据集类，从网格中提取特定部分。
    固定为1行3列格式。
    每张小图像大小为512x512，网格宽度固定为1536px。
    第1列是target image，第2列是ref image，第3列是source image。
    返回source、target、refer三张图像。

    Args:
        dir_list: 目录列表，如 [dir_a, dir_b]，包含图像文件
    """

    def __init__(self, dir_list):
        self.dir_list = dir_list

        # 从目录列表中加载所有图像文件
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

        # 网格配置: 3列（固定），1行
        self.grid_cols = 3
        self.grid_rows = 1  # 固定为1行
        self.tile_width = 512  # 小图像宽度 (从1024改为512)
        self.tile_height = 512  # 小图像高度 (从1024改为512)
        self.image_width = self.tile_width * self.grid_cols  # 1536 (从3072改为1536)

        # 定义数据增强变换
        self.norm = transforms.Normalize([0.5], [0.5])
        self.to_tensor = transforms.ToTensor()
        prob = 0.7
        # Source和target的增强变换（简化版）
        self.pixel_transform = A.Compose(
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

        # Refer图像的增强变换（简化版）
        self.hair_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=512),
                A.CenterCrop(512, 512),
                A.Affine(scale=(0.9, 1.2), rotate=(-10, 10), p=0.7),
            ]
        )

        print(f"Loaded {len(self.image_paths)} images from directories: {dir_list}")

    def refer_imgaug(self, image):
        image = image[:512, :512]
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), [512, 512])
        results = self.hair_transform(image=image)
        image = self.norm(self.to_tensor(results["image"] / 255.0))
        return image

    def imgaug(self, source_image, target_image):
        source_image = source_image[:512, :512]
        target_image = target_image[:512, :512]
        source_image = cv2.resize(
            cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB), [512, 512]
        )
        target_image = cv2.resize(
            cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB), [512, 512]
        )
        results = self.pixel_transform(image=source_image, image0=target_image)
        source_image, target_image = (
            self.norm(self.to_tensor(results["image"] / 255.0)),
            self.norm(self.to_tensor(results["image0"] / 255.0)),
        )
        return source_image, target_image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img_total = cv2.imread(image_path)

        # 检查图像是否成功加载
        if img_total is None:
            raise ValueError(f"无法加载图像: {image_path}")

        # 固定1行格式，设置裁剪区域
        top = 0
        bottom = self.tile_height

        # 提取target图像（第1列，索引0）
        left_target = 0 * self.tile_width
        right_target = left_target + self.tile_width
        target_image = img_total[top:bottom, left_target:right_target]

        # 提取refer图像（第2列，索引1）
        left_refer = 1 * self.tile_width
        right_refer = left_refer + self.tile_width
        refer_image = img_total[top:bottom, left_refer:right_refer]

        # 提取source图像（第3列，索引2）
        left_source = 2 * self.tile_width
        right_source = left_source + self.tile_width
        source_image = img_total[top:bottom, left_source:right_source]

        # 应用数据增强变换
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
    # 创建数据集实例

    dset = GridImageDataset(
        [
            path_win2wsl(r"F:\dataset\select_clein_lq_r"),
            # r'F:\dataset\concat_output2'
        ]
    )

    # 随机选择一个样本进行展示
    import random
    import matplotlib.pyplot as plt
    import numpy as np

    # 随机选择一个索引
    sample_idx = random.randint(0, len(dset) - 1)
    print(f"\n正在展示第 {sample_idx} 个样本（共 {len(dset)} 个样本）")
    print("数据格式：1行3列，每张图像512x512像素")
    print("列顺序：第1列=target，第2列=ref，第3列=source")

    # 获取样本
    sample = dset[0]

    # 打印样本信息
    print(f"Source image shape: {sample['source_pixel_values'].shape}")
    print(f"Target image shape: {sample['target_pixel_values'].shape}")
    print(f"Refer image shape: {sample['refer_pixel_values'].shape}")

    # 设置matplotlib支持中文字体
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 可视化图像
    # 反归一化：将[-1, 1]范围的数据转换为[0, 1]范围
    def denormalize(tensor):
        return (tensor * 0.5 + 0.5).clamp(0, 1)

    # 提取图像并转换为numpy数组
    source_img = denormalize(sample["source_pixel_values"]).permute(1, 2, 0).numpy()
    target_img = denormalize(sample["target_pixel_values"]).permute(1, 2, 0).numpy()
    refer_img = denormalize(sample["refer_pixel_values"]).permute(1, 2, 0).numpy()

    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示图像
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
