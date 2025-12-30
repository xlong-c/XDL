# -*- coding: utf-8 -*-
import os
import glob

import albumentations as A
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Hair10HairDataset(Dataset):
    """
    从三个文件夹加载图像的数据集类。
    文件命名规则：
    - bimg_fold (source): {cls_id}_{index}.png
    - zimg_fold (target): {cls_id}_{index}.png (与source同名)
    - rimg_fold (refer): {cls_id}.png (只有cls_id，没有index)

    Args:
        bimg_fold: source图像文件夹路径
        zimg_fold: target图像文件夹路径
        rimg_fold: refer图像文件夹路径
    """

    def __init__(self, bimg_fold, zimg_fold, rimg_fold):
        self.bimg_fold = bimg_fold
        self.zimg_fold = zimg_fold
        self.rimg_fold = rimg_fold

        # 收集所有样本
        self.samples = []
        self._collect_samples()

        # 定义数据增强变换
        self.norm = transforms.Normalize([0.5], [0.5])
        self.to_tensor = transforms.ToTensor()
        prob = 0.7

        # Source和target的增强变换
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

        # Refer图像的增强变换
        self.hair_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=512),
                A.CenterCrop(512, 512),
                A.Affine(scale=(0.9, 1.2), rotate=(-10, 10), p=0.7),
            ]
        )

        print(f"Loaded {len(self.samples)} samples from {bimg_fold}")

    def _collect_samples(self):
        """
        扫描bimg文件夹，收集所有有效的样本。
        对于每个 {cls_id}_{index}.png 文件，验证对应的target和refer文件是否存在。
        """
        # 获取bimg文件夹中所有png文件
        bimg_files = glob.glob(os.path.join(self.bimg_fold, "*.png"))

        for bimg_path in bimg_files:
            # 获取文件名（不含路径和扩展名）
            filename = os.path.basename(bimg_path)
            basename = os.path.splitext(filename)[0]

            # 解析cls_id和index
            # 文件名格式：{cls_id}_{index}
            if "_" not in basename:
                print(f"Warning: Skipping invalid filename {filename} (no underscore)")
                continue

            parts = basename.rsplit("_", 1)  # 从右边分割，只分割一次
            if len(parts) != 2:
                print(f"Warning: Skipping invalid filename {filename}")
                continue

            cls_id, index = parts

            # 构造target和refer的文件路径
            target_path = os.path.join(self.zimg_fold, filename)
            refer_path = os.path.join(self.rimg_fold, f"{cls_id}.png")

            # 验证文件是否存在
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

            # 保存样本信息
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

    def refer_imgaug(self, image):
        """对refer图像应用变换"""
        image = cv2.resize(image, [512, 512])
        # 如果图像是BGR格式，转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hair_transform(image=image)
        image = self.norm(self.to_tensor(results["image"] / 255.0))
        return image

    def imgaug(self, source_image, target_image):
        """对source和target图像应用变换"""
        # 调整大小
        source_image = cv2.resize(source_image, [512, 512])
        target_image = cv2.resize(target_image, [512, 512])

        # 如果图像是BGR格式，转换为RGB
        if len(source_image.shape) == 3 and source_image.shape[2] == 3:
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        if len(target_image.shape) == 3 and target_image.shape[2] == 3:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        results = self.pixel_transform(image=source_image, image0=target_image)
        source_image, target_image = (
            self.norm(self.to_tensor(results["image"] / 255.0)),
            self.norm(self.to_tensor(results["image0"] / 255.0)),
        )
        return source_image, target_image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # 读取三张图像
        source_image = cv2.imread(sample_info["source_path"])
        target_image = cv2.imread(sample_info["target_path"])
        refer_image = cv2.imread(sample_info["refer_path"])

        # 检查图像是否成功加载
        if source_image is None:
            raise ValueError(f"无法加载source图像: {sample_info['source_path']}")
        if target_image is None:
            raise ValueError(f"无法加载target图像: {sample_info['target_path']}")
        if refer_image is None:
            raise ValueError(f"无法加载refer图像: {sample_info['refer_path']}")

        # 应用数据增强变换
        source_image, target_image = self.imgaug(source_image, target_image)
        refer_image = self.refer_imgaug(refer_image)

        batch = {
            "source_pixel_values": source_image,
            "target_pixel_values": target_image,
            "refer_pixel_values": refer_image,
        }
        return batch


if __name__ == "__main__":
    # 创建数据集实例
    # 请根据实际情况修改路径
    bimg_fold = r"F:\dataset\10hair\bimg"
    zimg_fold = r"F:\dataset\10hair\zimg"
    rimg_fold = r"F:\dataset\10hair\rimg"

    try:
        dset = Hair10HairDataset(bimg_fold, zimg_fold, rimg_fold)

        # 随机选择一个样本进行展示
        import random

        import matplotlib.pyplot as plt

        sample_idx = random.randint(0, len(dset) - 1)
        print(f"\n正在展示第 {sample_idx} 个样本(共 {len(dset)} 个样本)")
        print("数据格式：从三个独立文件夹加载source、target、refer图像")

        # 获取样本
        sample = dset[sample_idx]

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
        source_img = (
            denormalize(sample["source_pixel_values"]).permute(1, 2, 0).numpy()
        )
        target_img = (
            denormalize(sample["target_pixel_values"]).permute(1, 2, 0).numpy()
        )
        refer_img = denormalize(sample["refer_pixel_values"]).permute(1, 2, 0).numpy()

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 显示图像
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
