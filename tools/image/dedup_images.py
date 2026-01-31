#!/usr/bin/env python3
import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("警告: PIL 未安装，将使用文件哈希")
    print("安装 PIL: pip install Pillow")

try:
    from skimage.metrics import structural_similarity as ssim
    import numpy as np

    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("警告: scikit-image 未安装，SSIM 功能不可用")
    print("安装 scikit-image: pip install scikit-image")


class DuplicateImageFinder:
    def __init__(
        self,
        folder_path: str,
        hash_method: str = "perceptual",
        similarity_threshold: int = 5,
        workers: int = None,
        recursive: bool = True,
        sort_by: str = "size",  # "size"=按文件大小, "name"=按文件名
        batch_size: int = 500,  # 感知哈希模式的批处理大小
    ):
        self.folder_path = Path(folder_path)
        self.hash_method = hash_method
        self.similarity_threshold = similarity_threshold
        self.workers = workers or os.cpu_count()
        self.recursive = recursive
        self.sort_by = sort_by
        self.batch_size = batch_size
        self.image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".tiff",
            ".webp",
        }

    def get_all_images(self) -> List[Path]:
        images = []
        glob_func = self.folder_path.rglob if self.recursive else self.folder_path.glob
        for ext in self.image_extensions:
            images.extend(glob_func(f"*{ext}"))
            images.extend(glob_func(f"*{ext.upper()}"))
        return images

    def get_file_size(self, file_path: Path) -> int:
        try:
            return file_path.stat().st_size
        except OSError:
            return -1

    def group_by_size(self, images: List[Path]) -> Dict[int, List[Path]]:
        size_groups = defaultdict(list)
        for img in images:
            size = self.get_file_size(img)
            if size > 0:
                size_groups[size].append(img)

        potential_duplicates = {
            size: files for size, files in size_groups.items() if len(files) > 1
        }

        duplicate_count = sum(len(files) for files in potential_duplicates.values())
        print(
            f"  发现 {len(potential_duplicates)} 个大小组包含 {duplicate_count} 个潜在重复文件"
        )

        return potential_duplicates

    def compute_file_hash(
        self, file_path: Path, algorithm: str = "md5"
    ) -> Optional[str]:
        try:
            hasher = hashlib.md5() if algorithm == "md5" else hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None

    def compute_perceptual_hash(
        self, file_path: Path, hash_size: int = 8
    ) -> Optional[str]:
        if not HAS_PIL:
            return self.compute_file_hash(file_path)

        try:
            with Image.open(file_path) as img:
                img = img.convert("L")
                img = img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)

                pixels = list(img.getdata())

                diff = []
                for row in range(hash_size):
                    for col in range(hash_size):
                        left_pixel = pixels[row * (hash_size + 1) + col]
                        right_pixel = pixels[row * (hash_size + 1) + col + 1]
                        diff.append(left_pixel > right_pixel)

                decimal_value = sum(bit << i for i, bit in enumerate(diff))
                return format(decimal_value, f"0{hash_size * hash_size // 4}x")

        except Exception:
            return None

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        if len(hash1) != len(hash2):
            return 999999
        x = int(hash1, 16) ^ int(hash2, 16)
        return bin(x).count("1")

    def compute_ssim(self, img_path1: Path, img_path2: Path) -> float:
        """计算两张图片的SSIM相似度,返回0-1之间的值"""
        if not HAS_PIL or not HAS_SSIM:
            return 0.0

        try:
            with Image.open(img_path1) as img1, Image.open(img_path2) as img2:
                # 转换为RGB模式
                img1 = img1.convert("RGB")
                img2 = img2.convert("RGB")

                # resize到256x256
                img1 = img1.resize((256, 256), Image.Resampling.LANCZOS)
                img2 = img2.resize((256, 256), Image.Resampling.LANCZOS)

                # 转换为numpy数组
                arr1 = np.array(img1)
                arr2 = np.array(img2)

                # 计算SSIM
                score = ssim(arr1, arr2, channel_axis=2)
                return score
        except Exception:
            return 0.0

    def compute_hashes_parallel(self, files: List[Path]) -> Dict[Path, str]:
        results = {}

        if self.hash_method == "perceptual" and HAS_PIL:
            hash_func = self.compute_perceptual_hash
        else:

            def hash_func(file_path):
                return self.compute_file_hash(file_path, self.hash_method)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_file = {executor.submit(hash_func, f): f for f in files}

            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    hash_value = future.result()
                    if hash_value:
                        results[file_path] = hash_value
                except Exception:
                    pass

        return results

    def group_by_hash(self, files: List[Path]) -> Dict[str, List[Path]]:
        file_hashes = self.compute_hashes_parallel(files)

        hash_groups = defaultdict(list)
        for file_path, hash_value in file_hashes.items():
            hash_groups[hash_value].append(file_path)

        return hash_groups

    def load_and_resize(self, file_path: Path) -> Optional[np.ndarray]:
        """加载图片并resize到256x256,返回numpy数组"""
        if not HAS_PIL:
            return None
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                return np.array(img)
        except Exception:
            return None

    def find_similar_images_ssim(
        self, files: List[Path], ssim_threshold: float = 0.95
    ) -> List[List[Path]]:
        """使用SSIM算法查找相似图片

        逻辑:
        1. 按文件大小从大到小排序
        2. for循环遍历,维护temp缓存上一张resize后的图片
        3. 比较当前图片和temp的SSIM值
        4. 如果相似度超过阈值,将当前图片加入当前组
        5. 将当前图片resize后存入temp,继续下一次比对
        """
        if not HAS_PIL or not HAS_SSIM:
            print("  警告: PIL或scikit-image未安装,无法使用SSIM")
            return []

        if len(files) < 2:
            return []

        # 根据配置选择排序方式
        import re

        def natural_name_key(f: Path):
            """按文件名自然排序"""
            name = f.name
            parts = re.split(r"(\d+)", name)
            return [int(part) if part.isdigit() else part.lower() for part in parts]

        if self.sort_by == "name":
            sorted_files = sorted(files, key=natural_name_key)
            print(f"  排序方式: 按文件名（自然排序）")
        else:
            def size_then_name_key(f: Path):
                size = self.get_file_size(f)
                return (-size, natural_name_key(f))

            sorted_files = sorted(files, key=size_then_name_key)
            print(f"  排序方式: 按文件大小（降序）")

        similar_groups = []
        current_group = [sorted_files[0]]

        # 使用多线程并行加载所有图片
        print(f"  并行加载 {len(sorted_files)} 张图片...")
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_file = {executor.submit(self.load_and_resize, f): f for f in sorted_files}
            loaded_images = {}
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    img_array = future.result()
                    loaded_images[file_path] = img_array
                except Exception as e:
                    loaded_images[file_path] = None

        # 获取第一张图片
        temp_img = loaded_images[sorted_files[0]]
        if temp_img is None:
            print(f"    无法加载: {sorted_files[0].name}")
            return []

        print(f"  已加载完成,开始比较...")
        for i in range(1, len(sorted_files)):
            current_file = sorted_files[i]
            current_img = loaded_images[current_file]

            if current_img is None:
                if len(current_group) > 1:
                    similar_groups.append(current_group)
                current_group = []
                temp_img = None
                continue

            print(f"  比较: {sorted_files[i - 1].name} <-> {current_file.name}")
            try:
                ssim_score = ssim(temp_img, current_img, channel_axis=2)
            except Exception as e:
                print(f"    SSIM计算失败: {e}")
                ssim_score = 0.0

            if ssim_score >= ssim_threshold:
                print(f"    SSIM {ssim_score:.4f}: 相似,加入当前组")
                current_group.append(current_file)
            else:
                print(f"    SSIM {ssim_score:.4f}: 不相似")
                if len(current_group) > 1:
                    similar_groups.append(current_group)
                current_group = [current_file]

            temp_img = current_img

        if len(current_group) > 1:
            similar_groups.append(current_group)

        return similar_groups

    def find_duplicates(self) -> List[List[Path]]:
        print(f"\n{'=' * 60}")
        print(f"扫描文件夹: {self.folder_path}")
        print(f"递归扫描: {'是' if self.recursive else '否'}")
        print(f"哈希方法: {self.hash_method}")
        print(f"工作进程: {self.workers}")
        print(f"{'=' * 60}")

        images = self.get_all_images()
        if not images:
            print("未找到图片文件")
            return []

        print(f"找到 {len(images)} 个图片文件")

        # SSIM模式跳过按大小分组
        if self.hash_method == "ssim":
            print("\n[阶段1/2] 按文件大小排序并准备SSIM检测...")
            return self.find_duplicates_ssim(images)

        print("\n[阶段1/3] 按文件大小分组（流式处理）...")

        size_groups = self.group_by_size(images)
        all_duplicates = []

        # 阶段2: 计算精确重复（相同大小+相同哈希）- 流式处理每个大小组
        if size_groups:
            print("\n[阶段2/3] 计算文件哈希值（流式处理）...")

            total_groups = len(size_groups)
            for idx, (size, files) in enumerate(size_groups.items(), 1):
                print(
                    f"  处理组 {idx}/{total_groups} (大小: {size:,} bytes, 文件数: {len(files)})",
                    end="\r",
                )

                hash_groups = self.group_by_hash(files)

                for hash_value, file_list in hash_groups.items():
                    if len(file_list) > 1:
                        all_duplicates.append(file_list)

            print(f"\n  发现 {len(all_duplicates)} 组精确重复")
        else:
            print("\n[阶段2/3] 未发现相同大小的文件，跳过精确重复检测")

        # 阶段3: 感知哈希模式下检测相似图片（无论是否有精确重复都执行）
        if self.hash_method == "perceptual" and self.similarity_threshold > 0:
            print("\n[阶段3/3] 使用感知哈希检测相似图片（批处理）...")
            similar_groups = self.find_similar_images_perceptual(
                images, self.similarity_threshold
            )
            if similar_groups:
                print(f"  发现 {len(similar_groups)} 组相似图片")
                all_duplicates.extend(similar_groups)

        return all_duplicates

    def find_duplicates_ssim(self, images: List[Path]) -> List[List[Path]]:
        """SSIM模式查找重复图片"""
        similar_groups = self.find_similar_images_ssim(
            images, self.similarity_threshold
        )
        if similar_groups:
            print(f"\n  发现 {len(similar_groups)} 组相似图片")
        return similar_groups

    def find_similar_images_perceptual(
        self, files: List[Path], threshold: int = 10
    ) -> List[List[Path]]:
        """使用感知哈希查找相似图片（流式处理）

        逻辑:
        1. 分批处理图片，每批 batch_size 张
        2. 每批内按排序方式排序
        3. 计算每批的感知哈希（使用多线程加速）
        4. 比较批内相邻图片的汉明距离
        5. 距离小于阈值的认为是相似图片
        """
        if not HAS_PIL:
            print("  警告: PIL未安装,无法使用感知哈希")
            return []

        if len(files) < 2:
            return []

        all_similar_groups = []
        import re

        def natural_name_key(f: Path):
            """按文件名自然排序"""
            name = f.name
            parts = re.split(r"(\d+)", name)
            return [int(part) if part.isdigit() else part.lower() for part in parts]

        # 排序所有文件
        if self.sort_by == "name":
            sorted_files = sorted(files, key=natural_name_key)
            print(f"  排序方式: 按文件名（自然排序）")
        else:
            def size_then_name_key(f: Path):
                size = self.get_file_size(f)
                return (-size, natural_name_key(f))
            sorted_files = sorted(files, key=size_then_name_key)
            print(f"  排序方式: 按文件大小（降序）")

        # 分批处理
        total_files = len(sorted_files)
        total_batches = (total_files + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_files)
            batch_files = sorted_files[start_idx:end_idx]

            print(f"\n  批次 {batch_idx + 1}/{total_batches}: 处理 {len(batch_files)} 张图片")

            if len(batch_files) < 2:
                continue

            # 计算批内文件的感知哈希（使用多线程加速）
            print(f"    计算哈希...")
            file_hashes = self.compute_hashes_parallel(batch_files)

            if len(file_hashes) < 2:
                continue

            # 批内比较汉明距离
            current_group = [batch_files[0]]

            for i in range(1, len(batch_files)):
                file1 = batch_files[i - 1]
                file2 = batch_files[i]

                if file1 not in file_hashes or file2 not in file_hashes:
                    if len(current_group) > 1:
                        all_similar_groups.append(current_group)
                    current_group = [file2]
                    continue

                hash1 = file_hashes[file1]
                hash2 = file_hashes[file2]

                distance = self.hamming_distance(hash1, hash2)
                print(f"    比较: {file1.name} <-> {file2.name} (汉明距离: {distance})")

                if distance <= threshold:
                    print(f"      相似 (距离 {distance} <= {threshold})")
                    current_group.append(file2)
                else:
                    print(f"      不相似 (距离 {distance} > {threshold})")
                    if len(current_group) > 1:
                        all_similar_groups.append(current_group)
                    current_group = [file2]

            # 保存批内最后一组
            if len(current_group) > 1:
                all_similar_groups.append(current_group)

        return all_similar_groups

    def report_duplicates(self, duplicates: List[List[Path]], output_file: str = None):
        if not duplicates:
            print("\n未发现重复图片！")
            return

        total_duplicates = sum(len(group) - 1 for group in duplicates)
        total_wasted_space = 0

        print(f"\n{'=' * 60}")
        print("重复图片检测报告")
        print(f"{'=' * 60}")
        print(f"发现 {len(duplicates)} 组重复，共 {total_duplicates} 个重复文件\n")

        report_lines = []

        for i, group in enumerate(duplicates, 1):
            group_size = self.get_file_size(group[0])
            wasted = group_size * (len(group) - 1)
            total_wasted_space += wasted

            print(f"组 {i} ({len(group)} 个文件, 每组大小: {group_size:,} bytes):")
            report_lines.append(f"\n组 {i}:")

            for j, file_path in enumerate(group):
                prefix = "  [保留]" if j == 0 else "  [重复]"
                print(f"  {prefix} {file_path}")
                report_lines.append(f"{prefix} {file_path}")

            print()

        print(f"{'=' * 60}")
        print(f"预计可释放空间: {self.format_size(total_wasted_space)}")
        print(f"{'=' * 60}")

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"\n报告已保存到: {output_file}")

    def format_size(self, size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"

    def remove_duplicates(
        self,
        duplicates: List[List[Path]],
        keep_first: bool = True,
        dry_run: bool = True,
    ) -> Tuple[int, int]:
        if not duplicates:
            return 0, 0

        deleted_count = 0
        freed_space = 0

        action = "将删除" if dry_run else "已删除"

        print(f"\n{'=' * 60}")
        print(f"{'[模拟运行]' if dry_run else '[实际删除]'} 删除重复图片")
        print(f"{'=' * 60}")

        for group in duplicates:
            if keep_first:
                to_keep = group[0]
                to_delete = group[1:]
            else:
                to_keep = max(group, key=lambda p: p.stat().st_mtime)
                to_delete = [f for f in group if f != to_keep]

            print(f"\n保留: {to_keep}")

            for file_path in to_delete:
                file_size = self.get_file_size(file_path)
                print(f"  {action}: {file_path}")

                if not dry_run:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        freed_space += file_size
                    except Exception as e:
                        print(f"  删除失败: {e}")
                else:
                    deleted_count += 1
                    freed_space += file_size

        print(f"\n{'=' * 60}")
        print(f"总计: {action} {deleted_count} 个文件")
        print(f"释放空间: {self.format_size(freed_space)}")
        print(f"{'=' * 60}")

        return deleted_count, freed_space


def main():
    # 哈希模式: "md5"=精确匹配, "perceptual"=感知哈希, "ssim"=SSIM相似度检测
    HASH_METHOD = "perceptual"
    # 感知哈希相似度阈值(汉明距离,越小越相似),建议5-15 (基于8x8哈希,总位数为64)
    SIMILARITY_THRESHOLD = 15
    # 排序方式: "size"=按文件大小(降序), "name"=按文件名(自然排序)
    SORT_BY = "name"
    # 图片文件夹路径
    folder = "/mnt/f/raw_pics/good/001"
    delete_duplicates = True  # 是否执行删除操作
    dry_run = False  # True=仅预览不删除, False=实际删除
    keep_latest = True  # True=保留最新修改的文件, False=保留每组第一个
    workers = 12  # 并行进程数, None=使用CPU核心数
    recursive = True  # True=递归扫描子文件夹, False=仅扫描当前文件夹
    batch_size = 500  # 感知哈希模式的批处理大小（流式处理，减少内存占用）

    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"错误: 文件夹不存在: {folder}")
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"错误: 不是文件夹: {folder}")
        sys.exit(1)

    finder = DuplicateImageFinder(
        folder_path=str(folder_path),
        hash_method=HASH_METHOD,
        similarity_threshold=SIMILARITY_THRESHOLD,
        workers=workers,
        recursive=recursive,
        sort_by=SORT_BY,
        batch_size=batch_size,
    )

    duplicates = finder.find_duplicates()

    finder.report_duplicates(duplicates)

    if delete_duplicates or dry_run:
        finder.remove_duplicates(
            duplicates, keep_first=not keep_latest, dry_run=dry_run
        )


if __name__ == "__main__":
    main()
