#!/usr/bin/env python3
import sys
import hashlib
import re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def path_win2wsl(win_path: str) -> str:
    if not win_path:
        return win_path
    win_path = win_path.strip()
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path


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
except ImportError as e:
    HAS_SSIM = False
    print(f"警告: scikit-image 未安装，SSIM 功能不可用: {e}")
    print("安装方式:")
    print("  方法1: pip install scikit-image")
    print("  方法2: pip install xdl[cv]")
    print("  方法3: conda install scikit-image")


def find_images(folder_path: Path, recursive: bool = True):
    glob_func = folder_path.rglob if recursive else folder_path.glob
    for ext in IMAGE_EXTENSIONS:
        for p in glob_func(f"*{ext}"):
            yield p.relative_to(folder_path), p
        for p in glob_func(f"*{ext.upper()}"):
            yield p.relative_to(folder_path), p


def get_file_size(file_path: Path) -> int:
    try:
        return file_path.stat().st_size
    except OSError:
        return -1


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def compute_file_hash(file_path: Path, algorithm: str = "md5") -> Optional[str]:
    try:
        hasher = hashlib.md5() if algorithm == "md5" else hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def compute_perceptual_hash(file_path: Path, hash_size: int = 8) -> Optional[str]:
    if not HAS_PIL:
        return compute_file_hash(file_path)

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


def hamming_distance(hash1: str, hash2: str) -> int:
    if len(hash1) != len(hash2):
        return 999999
    x = int(hash1, 16) ^ int(hash2, 16)
    return bin(x).count("1")


def load_and_resize(file_path: Path) -> Optional[np.ndarray]:
    if not HAS_PIL:
        return None
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            return np.array(img)
    except Exception:
        return None


def compute_ssim(img_path1: Path, img_path2: Path) -> float:
    if not HAS_PIL or not HAS_SSIM:
        return 0.0

    try:
        with Image.open(img_path1) as img1, Image.open(img_path2) as img2:
            img1 = img1.convert("RGB")
            img2 = img2.convert("RGB")
            img1 = img1.resize((256, 256), Image.Resampling.LANCZOS)
            img2 = img2.resize((256, 256), Image.Resampling.LANCZOS)
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            score = ssim(arr1, arr2, channel_axis=2)
            return score
    except Exception:
        return 0.0


def natural_name_key(f: Path):
    name = f.name
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def size_then_name_key(f: Path):
    size = get_file_size(f)
    return (-size, natural_name_key(f))


def compute_hashes_parallel(
    files: List[Path], hash_method: str, workers: int
) -> Dict[Path, str]:
    results = {}

    if hash_method == "perceptual" and HAS_PIL:
        hash_func = compute_perceptual_hash
    else:

        def hash_func(file_path):
            return compute_file_hash(file_path, hash_method)

    with ThreadPoolExecutor(max_workers=workers) as executor:
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


def group_by_size(images: List[Path]) -> Dict[int, List[Path]]:
    size_groups = defaultdict(list)
    for img in images:
        size = get_file_size(img)
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


def group_by_hash(
    files: List[Path], hash_method: str, workers: int
) -> Dict[str, List[Path]]:
    file_hashes = compute_hashes_parallel(files, hash_method, workers)

    hash_groups = defaultdict(list)
    for file_path, hash_value in file_hashes.items():
        hash_groups[hash_value].append(file_path)

    return hash_groups


def find_similar_images_ssim(
    files: List[Path], ssim_threshold: float, sort_by: str, workers: int
) -> List[List[Path]]:
    if not HAS_PIL or not HAS_SSIM:
        print("  警告: PIL或scikit-image未安装,无法使用SSIM")
        return []

    if len(files) < 2:
        return []

    if sort_by == "name":
        sorted_files = sorted(files, key=natural_name_key)
        print("  排序方式: 按文件名(自然排序)")
    else:
        sorted_files = sorted(files, key=size_then_name_key)
        print("  排序方式: 按文件大小(降序)")

    print("  并行加载 {len(sorted_files)} 张图片...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {executor.submit(load_and_resize, f): f for f in sorted_files}
        loaded_images = {}
        for future in future_to_file:
            file_path = future_to_file[future]
            try:
                img_array = future.result()
                loaded_images[file_path] = img_array
            except Exception:
                loaded_images[file_path] = None

    temp_img = loaded_images[sorted_files[0]]
    if temp_img is None:
        print(f"    无法加载: {sorted_files[0].name}")
        return []

    similar_groups = []
    current_group = [sorted_files[0]]

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


def find_similar_images_perceptual(
    files: List[Path], threshold: int, sort_by: str, batch_size: int, workers: int
) -> List[List[Path]]:
    if not HAS_PIL:
        print("  警告: PIL未安装,无法使用感知哈希")
        return []

    if len(files) < 2:
        return []

    if sort_by == "name":
        sorted_files = sorted(files, key=natural_name_key)
        print(f"  排序方式: 按文件名(自然排序)")
    else:
        sorted_files = sorted(files, key=size_then_name_key)
        print(f"  排序方式: 按文件大小(降序)")

    total_files = len(sorted_files)
    total_batches = (total_files + batch_size - 1) // batch_size
    all_similar_groups = []

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_files = sorted_files[start_idx:end_idx]

        print(
            f"\n  批次 {batch_idx + 1}/{total_batches}: 处理 {len(batch_files)} 张图片"
        )

        if len(batch_files) < 2:
            continue

        print(f"    计算哈希...")
        file_hashes = compute_hashes_parallel(batch_files, "perceptual", workers)

        if len(file_hashes) < 2:
            continue

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
            distance = hamming_distance(hash1, hash2)

            print(f"    比较: {file1.name} <-> {file2.name} (汉明距离: {distance})")

            if distance <= threshold:
                print(f"      相似 (距离 {distance} <= {threshold})")
                current_group.append(file2)
            else:
                print(f"      不相似 (距离 {distance} > {threshold})")
                if len(current_group) > 1:
                    all_similar_groups.append(current_group)
                current_group = [file2]

        if len(current_group) > 1:
            all_similar_groups.append(current_group)

    return all_similar_groups


def find_duplicates(
    folder_path: Path,
    hash_method: str,
    similarity_threshold: int,
    recursive: bool,
    sort_by: str,
    batch_size: int,
    workers: int,
) -> List[List[Path]]:
    print(f"\n{'=' * 60}")
    print(f"扫描文件夹: {folder_path}")
    print(f"递归扫描: {'是' if recursive else '否'}")
    print(f"哈希方法: {hash_method}")
    print(f"工作进程: {workers}")
    print(f"{'=' * 60}")

    images = list(find_images(folder_path, recursive))
    if not images:
        print("未找到图片文件")
        return []

    print(f"找到 {len(images)} 个图片文件")

    if hash_method == "ssim":
        print("\n[阶段1/2] 按文件大小排序并准备SSIM检测...")
        files = [p for _, p in images]
        similar_groups = find_similar_images_ssim(
            files, similarity_threshold / 100.0, sort_by, workers
        )
        if similar_groups:
            print(f"\n  发现 {len(similar_groups)} 组相似图片")
        return similar_groups

    print("\n[阶段1/3] 按文件大小分组(流式处理)...")
    files = [p for _, p in images]
    size_groups = group_by_size(files)
    all_duplicates = []

    if size_groups:
        print("\n[阶段2/3] 计算文件哈希值(流式处理)...")
        total_groups = len(size_groups)
        for idx, (size, files_in_group) in enumerate(size_groups.items(), 1):
            print(
                f"  处理组 {idx}/{total_groups} (大小: {size:,} bytes, 文件数: {len(files_in_group)})",
                end="\r",
            )
            hash_groups = group_by_hash(files_in_group, hash_method, workers)
            for hash_value, file_list in hash_groups.items():
                if len(file_list) > 1:
                    all_duplicates.append(file_list)
        print(f"\n  发现 {len(all_duplicates)} 组精确重复")
    else:
        print("\n[阶段2/3] 未发现相同大小的文件,跳过精确重复检测")

    if hash_method == "perceptual" and similarity_threshold > 0:
        print("\n[阶段3/3] 使用感知哈希检测相似图片(批处理)...")
        similar_groups = find_similar_images_perceptual(
            files, similarity_threshold, sort_by, batch_size, workers
        )
        if similar_groups:
            print(f"  发现 {len(similar_groups)} 组相似图片")
            all_duplicates.extend(similar_groups)

    return all_duplicates


def preview_duplicates(
    folder_path: Path,
    duplicates: List[List[Path]],
    hash_method: str,
    similarity_threshold: int,
    sort_by: str,
    preview_limit: int = 10,
) -> bool:
    if not duplicates:
        print("\n未发现重复图片!")
        return False

    print(f"\n{'=' * 80}")
    print(f"预览模式 (共发现 {len(duplicates)} 组重复, 显示前 {preview_limit} 组):")
    print(f"{'=' * 80}")

    for i, group in enumerate(duplicates[:preview_limit], 1):
        group_size = get_file_size(group[0])
        print(f"\n[操作: 重复图片检测和清理]")
        print(f"  源: {folder_path}")
        print(f"  到: {folder_path}")
        print(
            f"  参数: hash={hash_method}, threshold={similarity_threshold}, sort={sort_by}"
        )
        print(f"  文件数: {len(group)} 个, 大小: {format_size(group_size)}")
        for j, file_path in enumerate(group):
            status = "保留" if j == 0 else "删除"
            print(f"    [{status}] {file_path}")

    if len(duplicates) > preview_limit:
        remaining = len(duplicates) - preview_limit
        remaining_files = sum(len(g) - 1 for g in duplicates[preview_limit:])
        print(f"\n... 还有 {remaining} 组 ({remaining_files} 个文件) ...")

    print(f"{'=' * 80}")
    user_input = input("\n确认继续? (y/n): ").lower()
    return user_input == "y"


def report_duplicates(duplicates: List[List[Path]], output_file: str = None):
    if not duplicates:
        print("\n未发现重复图片!")
        return

    total_duplicates = sum(len(group) - 1 for group in duplicates)
    total_wasted_space = 0

    print(f"\n{'=' * 60}")
    print("重复图片检测报告")
    print(f"{'=' * 60}")
    print(f"发现 {len(duplicates)} 组重复,共 {total_duplicates} 个重复文件\n")

    report_lines = []

    for i, group in enumerate(duplicates, 1):
        group_size = get_file_size(group[0])
        wasted = group_size * (len(group) - 1)
        total_wasted_space += wasted

        print(f"组 {i} ({len(group)} 个文件,每组大小: {group_size:,} bytes):")
        report_lines.append(f"\n组 {i}:")

        for j, file_path in enumerate(group):
            prefix = "  [保留]" if j == 0 else "  [重复]"
            print(f"  {prefix} {file_path}")
            report_lines.append(f"{prefix} {file_path}")

        print()

    print(f"{'=' * 60}")
    print(f"预计可释放空间: {format_size(total_wasted_space)}")
    print(f"{'=' * 60}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n报告已保存到: {output_file}")


def remove_duplicates(
    duplicates: List[List[Path]],
    keep_first: bool = True,
    dry_run: bool = True,
    output_path: Path = None,
) -> Tuple[int, int]:
    if not duplicates:
        return 0, 0

    deleted_count = 0
    freed_space = 0
    failed_files = []

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
            file_size = get_file_size(file_path)
            print(f"  {action}: {file_path}")

            if not dry_run:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    freed_space += file_size
                except Exception as e:
                    error_msg = f"删除失败: {e}"
                    print(f"  {error_msg}")
                    failed_files.append((file_path.name, error_msg))
            else:
                deleted_count += 1
                freed_space += file_size

    print(f"\n{'=' * 60}")
    print(f"总计: {action} {deleted_count} 个文件")
    print(f"释放空间: {format_size(freed_space)}")
    print(f"{'=' * 60}")

    if failed_files and output_path:
        log_path = output_path / "_delete_failed.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# 删除失败记录\n")
            f.write(f"# 总计: {len(failed_files)} 个文件失败\n\n")
            for name, error in failed_files:
                f.write(f"{name}: {error}\n")
        print(f"\n失败文件列表已保存到: {log_path}")

    return deleted_count, freed_space


def main():
    # ============ 配置参数 ============
    FOLDER = path_win2wsl(r"/mnt/f/raw_pics/good/001")
    HASH_METHOD = "perceptual"
    SIMILARITY_THRESHOLD = 15
    SORT_BY = "name"
    RECURSIVE = True
    WORKERS = 12
    BATCH_SIZE = 500
    PREVIEW = True
    PREVIEW_LIMIT = 10
    DELETE_DUPLICATES = True
    DRY_RUN = False
    KEEP_LATEST = True
    # ==================================

    folder_path = Path(FOLDER)
    if not folder_path.exists():
        print(f"错误: 文件夹不存在: {FOLDER}")
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"错误: 不是文件夹: {FOLDER}")
        sys.exit(1)

    duplicates = find_duplicates(
        folder_path=folder_path,
        hash_method=HASH_METHOD,
        similarity_threshold=SIMILARITY_THRESHOLD,
        recursive=RECURSIVE,
        sort_by=SORT_BY,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
    )

    if PREVIEW:
        if not preview_duplicates(
            folder_path=folder_path,
            duplicates=duplicates,
            hash_method=HASH_METHOD,
            similarity_threshold=SIMILARITY_THRESHOLD,
            sort_by=SORT_BY,
            preview_limit=PREVIEW_LIMIT,
        ):
            print("已取消")
            return

    report_duplicates(duplicates)

    if DELETE_DUPLICATES or DRY_RUN:
        remove_duplicates(
            duplicates,
            keep_first=not KEEP_LATEST,
            dry_run=DRY_RUN,
            output_path=folder_path,
        )


if __name__ == "__main__":
    main()
