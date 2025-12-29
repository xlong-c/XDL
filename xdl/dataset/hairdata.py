import csv
import os
import random

import albumentations as A
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class GridImageDataset(Dataset):
    """
    一个从CSV索引加载图像的数据集类, 从网格中提取特定部分。
    支持动态行数：2行、3行、4行等(3列固定)。
    每张小图像大小为768x1024, 网格宽度固定为2304px。
    第2列是source image, 第3列是target image(同一行), 第3列的其他行作为refer。
    返回source、target、refer三张图像。

    Args:
        csv_file: CSV文件路径, 包含图像文件列表
        base_dir: 图像基础目录
        grid_rows: 网格行数(可选)。如果未指定, 将自动检测每张图像的实际行数
    """

    def __init__(self, csv_file, base_dir="/root/autodl-tmp", grid_rows=None):
        self.csv_file = csv_file
        self.base_dir = base_dir

        # Load image paths from CSV
        self.image_paths = []
        with open(csv_file, encoding="utf-8") as f:
            print()
            reader = csv.reader(f)
            for row in reader:
                if row:  # Skip empty rows
                    self.image_paths.append(os.path.join(base_dir, row[0]))

        # 网格配置: 3列(固定), 行数动态
        self.grid_cols = 3
        self.grid_rows = grid_rows  # 动态设置行数
        self.tile_width = 768  # 小图像宽度
        self.tile_height = 1024  # 小图像高度
        self.image_width = self.tile_width * self.grid_cols  # 2304

        # 定义数据增强变换
        self.norm = transforms.Normalize([0.5], [0.5])
        self.to_tensor = transforms.ToTensor()
        prob = 0.7
        # Source和target的增强变换(简化版)
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

        # Refer图像的增强变换(简化版)
        self.hair_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=512),
                A.CenterCrop(512, 512),
                A.Affine(scale=(0.9, 1.2), rotate=(-10, 10), p=0.7),
            ]
        )

        print(f"Loaded {len(self.image_paths)} images from {csv_file}")

    def refer_imgaug(self, image):
        """对refer图像应用变换"""
        # 先裁剪顶部的768x768部分
        image = image[:768, :768]
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), [512, 512])
        results = self.hair_transform(image=image)
        image = self.norm(self.to_tensor(results["image"] / 255.0))
        return image

    def imgaug(self, source_image, target_image):
        """对source和target图像应用变换"""
        # 先裁剪顶部的768x768部分
        source_image = source_image[:768, :768]
        target_image = target_image[:768, :768]
        source_image = cv2.resize(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB), [512, 512])
        target_image = cv2.resize(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB), [512, 512])
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

        # 动态检测图像的行数
        actual_grid_rows = img_total.shape[0] // self.tile_height
        if self.grid_rows is None:
            # 如果构造函数未指定行数, 使用检测到的行数
            actual_grid_rows = img_total.shape[0] // self.tile_height
        else:
            # 如果指定了行数, 验证是否匹配
            if self.grid_rows != img_total.shape[0] // self.tile_height:
                print(
                    f"警告: 期望行数 {self.grid_rows} 与实际行数 {img_total.shape[0] // self.tile_height} 不匹配"
                )
            actual_grid_rows = self.grid_rows

        # 随机选择一行作为source
        selected_row = random.randint(0, actual_grid_rows - 1)

        # 提取source图像(第2列, 索引1)
        left_source = 1 * self.tile_width
        top = selected_row * self.tile_height
        right_source = left_source + self.tile_width
        bottom = top + self.tile_height

        source_image = img_total[top:bottom, left_source:right_source]

        # 提取target图像(第3列, 索引2, same row as source)
        left_target = 2 * self.tile_width
        right_target = left_target + self.tile_width

        target_image = img_total[top:bottom, left_target:right_target]

        # 从第3列中随机选择一个refer图像(排除source行)
        available_rows = [row for row in range(actual_grid_rows) if row != selected_row]
        refer_row = random.choice(available_rows)

        # 提取refer图像(第3列, 索引2, different row)
        left_refer = 2 * self.tile_width
        top_refer = refer_row * self.tile_height
        right_refer = left_refer + self.tile_width
        bottom_refer = top_refer + self.tile_height

        refer_image = img_total[top_refer:bottom_refer, left_refer:right_refer]

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
    def get_image_paths_from_csv(csv_file, base_dir="/root/autodl-tmp"):
        """
        从CSV文件获取图像路径
        """
        image_paths = []
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Skip empty rows
                    image_paths.append(os.path.join(base_dir, row[0]))
        return image_paths


if __name__ == "__main__":
    # 创建数据集实例
    dset = GridImageDataset(
        r"F:\BaiduNetDiskDownload\ture_data\newdata_1111.csv", r"F:\BaiduNetDiskDownload\ture_data"
    )

    # 随机选择一个样本进行展示
    import random

    import matplotlib.pyplot as plt

    # 随机选择一个索引
    sample_idx = random.randint(0, len(dset) - 1)
    print(f"\n正在展示第 {sample_idx} 个样本(共 {len(dset)} 个样本)")

    # 获取样本
    sample = dset[sample_idx]

    # 打印样本信息
    print(f"Source image shape: {sample['source_pixel_values'].shape}")
    print(f"Target image shape: {sample['target_pixel_values'].shape}")
    print(f"Refer image shape: {sample['refer_pixel_values'].shape}")

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
    axes[0].set_title("Source Image")
    axes[0].axis("off")

    axes[1].imshow(target_img)
    axes[1].set_title("Target Image")
    axes[1].axis("off")

    axes[2].imshow(refer_img)
    axes[2].set_title("Refer Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("\n样本展示完成！")
