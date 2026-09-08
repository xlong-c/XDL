import csv
import os
import random

import cv2
from torch.utils.data import Dataset

from .hair_transforms import HairAugMixin


class GridImageDataset(HairAugMixin, Dataset):
    """一个从CSV索引加载图像的数据集类, 从网格中提取特定部分。"""

    def __init__(self, csv_file: str, base_dir: str, grid_rows: int | None = None) -> None:
        self.csv_file = csv_file
        self.base_dir = base_dir

        self.image_paths = []
        with open(csv_file, encoding="utf-8") as f:
            print()
            reader = csv.reader(f)
            for row in reader:
                if row:
                    self.image_paths.append(os.path.join(base_dir, row[0]))

        self.grid_cols = 3
        self.grid_rows = grid_rows
        self.tile_width = 768
        self.tile_height = 1024
        self.image_width = self.tile_width * self.grid_cols

        self._init_hair_aug(crop_size=768)

        print(f"Loaded {len(self.image_paths)} images from {csv_file}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img_total = cv2.imread(image_path)

        if img_total is None:
            raise ValueError(f"无法加载图像: {image_path}")

        actual_grid_rows = img_total.shape[0] // self.tile_height
        if self.grid_rows is None:
            actual_grid_rows = img_total.shape[0] // self.tile_height
        else:
            if self.grid_rows != img_total.shape[0] // self.tile_height:
                print(
                    f"警告: 期望行数 {self.grid_rows} 与实际行数 {img_total.shape[0] // self.tile_height} 不匹配"
                )
            actual_grid_rows = self.grid_rows

        selected_row = random.randint(0, actual_grid_rows - 1)

        left_source = 1 * self.tile_width
        top = selected_row * self.tile_height
        right_source = left_source + self.tile_width
        bottom = top + self.tile_height

        source_image = img_total[top:bottom, left_source:right_source]

        left_target = 2 * self.tile_width
        right_target = left_target + self.tile_width

        target_image = img_total[top:bottom, left_target:right_target]

        available_rows = [row for row in range(actual_grid_rows) if row != selected_row]
        refer_row = random.choice(available_rows)

        left_refer = 2 * self.tile_width
        top_refer = refer_row * self.tile_height
        right_refer = left_refer + self.tile_width
        bottom_refer = top_refer + self.tile_height

        refer_image = img_total[top_refer:bottom_refer, left_refer:right_refer]

        source_image, target_image = self.imgaug(source_image, target_image)
        refer_image = self.refer_imgaug(refer_image)

        batch = {
            "source_pixel_values": source_image,
            "target_pixel_values": target_image,
            "refer_pixel_values": refer_image,
        }
        return batch

    @staticmethod
    def get_image_paths_from_csv(csv_file: str, base_dir: str) -> list[str]:
        image_paths = []
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    image_paths.append(os.path.join(base_dir, row[0]))
        return image_paths
