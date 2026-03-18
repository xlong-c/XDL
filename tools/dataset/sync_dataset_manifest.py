#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集同步脚本

用途:
1. collect 模式: 扫描本地数据集, 生成相对路径 CSV 清单。
2. clean 模式: 读取 CSV 清单, 删除目标数据集中不在清单里的文件。

说明:
- CSV 默认保存在当前 py 文件同级目录, 不保存在图片目录。
- CSV 只记录相对路径, 便于在不同机器之间同步清洗结果。
- 默认只处理图片文件; 如需处理所有文件, 可将 ONLY_IMAGE_FILES 改为 False。
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable

# 支持的图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def path_win2wsl(win_path: str) -> str:
    """将 Windows 路径转换为 WSL 路径"""
    if not win_path:
        return win_path

    clean_path = win_path.strip().replace('"', "").replace("'", "")
    if len(clean_path) >= 2 and clean_path[1] == ":":
        drive = clean_path[0].lower()
        rest = clean_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return clean_path


def normalize_relative_path(relative_path: Path | str) -> str:
    """统一相对路径格式, 便于跨平台比较"""
    text = str(relative_path).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if not text:
        return text
    return PurePosixPath(text).as_posix()


def should_include_file(file_path: Path, only_image_files: bool) -> bool:
    """判断文件是否应纳入扫描范围"""
    if not file_path.is_file():
        return False
    if only_image_files:
        return file_path.suffix.lower() in IMAGE_EXTENSIONS
    return True


def find_files(
    input_path: Path,
    only_image_files: bool,
    exclude_paths: Iterable[Path] | None = None,
) -> list[tuple[str, Path]]:
    """递归查找需要处理的文件"""
    input_path = input_path.resolve()
    excluded = {path.resolve() for path in (exclude_paths or [])}
    files: list[tuple[str, Path]] = []

    for root, _, filenames in os.walk(input_path):
        root_path = Path(root)
        for filename in filenames:
            abs_path = (root_path / filename).resolve()
            if abs_path in excluded:
                continue
            if not should_include_file(abs_path, only_image_files):
                continue

            relative_path = normalize_relative_path(abs_path.relative_to(input_path))
            files.append((relative_path, abs_path))

    files.sort(key=lambda item: item[0])
    return files


def write_manifest_csv(csv_path: Path, relative_paths: list[str]) -> None:
    """写入 CSV 清单"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["relative_path"])
        for relative_path in relative_paths:
            writer.writerow([relative_path])


def read_manifest_csv(csv_path: Path) -> set[str]:
    """读取 CSV 清单中的相对路径"""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    relative_paths: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            return relative_paths

        column_name = "relative_path"
        if column_name not in reader.fieldnames:
            column_name = reader.fieldnames[0]
            print(f"[!] 未找到 relative_path 列, 回退使用第一列: {column_name}")

        for row in reader:
            raw_value = row.get(column_name, "").strip()
            if raw_value:
                relative_paths.add(normalize_relative_path(raw_value))

    return relative_paths


def preview_collect(
    files: list[tuple[str, Path]],
    csv_path: Path,
    preview_limit: int,
    only_image_files: bool,
) -> None:
    """打印 collect 模式预览"""
    mode_text = "仅图片" if only_image_files else "所有文件"
    print("=" * 80)
    print("预览模式 [collect]")
    print("=" * 80)
    print(f"扫描类型: {mode_text}")
    print(f"总计文件: {len(files)}")
    print(f"CSV 输出: {csv_path}")
    print(f"显示前 {min(len(files), preview_limit)} 个条目")

    for index, (relative_path, abs_path) in enumerate(files[:preview_limit], 1):
        print("")
        print(f"[{index}]")
        print(f"  源: {abs_path}")
        print(f"  记录: {relative_path}")


def preview_clean(
    dataset_path: Path,
    extra_paths: list[str],
    missing_paths: list[str],
    csv_path: Path,
    preview_limit: int,
    only_image_files: bool,
) -> None:
    """打印 clean 模式预览"""
    mode_text = "仅图片" if only_image_files else "所有文件"
    print("=" * 80)
    print("预览模式 [clean]")
    print("=" * 80)
    print(f"目标目录: {dataset_path}")
    print(f"扫描类型: {mode_text}")
    print(f"CSV 清单: {csv_path}")
    print(f"待删除文件: {len(extra_paths)}")
    print(f"CSV 中存在但当前目录缺失: {len(missing_paths)}")

    if extra_paths:
        print("")
        print(f"待删除预览 (前 {min(len(extra_paths), preview_limit)} 个)")
        for index, relative_path in enumerate(extra_paths[:preview_limit], 1):
            print(f"  [{index}] {dataset_path / Path(relative_path)}")

    if missing_paths:
        print("")
        print(f"缺失文件预览 (前 {min(len(missing_paths), preview_limit)} 个)")
        for index, relative_path in enumerate(missing_paths[:preview_limit], 1):
            print(f"  [{index}] {relative_path}")


def confirm_execution() -> bool:
    """交互确认"""
    return input("\n确认执行? (y/n): ").strip().lower() == "y"


def save_failed_log(log_path: Path, failed_items: list[tuple[str, str]]) -> None:
    """保存失败日志"""
    if not failed_items:
        return

    with log_path.open("w", encoding="utf-8") as log_file:
        for relative_path, error in failed_items:
            log_file.write(f"{relative_path}: {error}\n")

    print(f"[!] 失败详情已记录至: {log_path}")


def save_deleted_log(log_path: Path, deleted_paths: list[str]) -> None:
    """保存删除记录"""
    if not deleted_paths:
        return

    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["deleted_relative_path"])
        for relative_path in deleted_paths:
            writer.writerow([relative_path])

    print(f"[*] 删除记录已保存至: {log_path}")


def remove_empty_directories(dataset_path: Path) -> int:
    """删除空目录"""
    removed_count = 0
    for root, dirnames, _ in os.walk(dataset_path, topdown=False):
        for dirname in dirnames:
            dir_path = Path(root) / dirname
            try:
                next(dir_path.iterdir())
            except StopIteration:
                dir_path.rmdir()
                removed_count += 1
            except Exception:
                continue
    return removed_count


def collect_manifest(
    dataset_path: Path,
    csv_path: Path,
    only_image_files: bool,
    preview: bool,
    preview_limit: int,
    exclude_paths: Iterable[Path] | None = None,
) -> None:
    """生成数据集清单 CSV"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"数据集路径不是目录: {dataset_path}")
    dataset_path = dataset_path.resolve()

    files = find_files(
        input_path=dataset_path,
        only_image_files=only_image_files,
        exclude_paths=exclude_paths,
    )
    relative_paths = [relative_path for relative_path, _ in files]

    if preview:
        preview_collect(
            files=files,
            csv_path=csv_path,
            preview_limit=preview_limit,
            only_image_files=only_image_files,
        )
        print("")
        if not confirm_execution():
            print("[!] 已取消 collect")
            return

    write_manifest_csv(csv_path=csv_path, relative_paths=relative_paths)
    print(f"[*] 已写入 CSV: {csv_path}")
    print(f"[*] 共记录 {len(relative_paths)} 个文件")


def clean_dataset(
    dataset_path: Path,
    csv_path: Path,
    only_image_files: bool,
    preview: bool,
    preview_limit: int,
    delete_empty_dirs: bool,
    exclude_paths: Iterable[Path] | None = None,
) -> None:
    """根据 CSV 清单清理多余文件"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"数据集路径不是目录: {dataset_path}")
    dataset_path = dataset_path.resolve()

    manifest_paths = read_manifest_csv(csv_path)
    current_files = find_files(
        input_path=dataset_path,
        only_image_files=only_image_files,
        exclude_paths=exclude_paths,
    )
    current_rel_paths = [relative_path for relative_path, _ in current_files]
    current_rel_path_set = set(current_rel_paths)

    extra_paths = sorted(current_rel_path_set - manifest_paths)
    missing_paths = sorted(manifest_paths - current_rel_path_set)

    if preview:
        preview_clean(
            dataset_path=dataset_path,
            extra_paths=extra_paths,
            missing_paths=missing_paths,
            csv_path=csv_path,
            preview_limit=preview_limit,
            only_image_files=only_image_files,
        )
        print("")
        if not confirm_execution():
            print("[!] 已取消 clean")
            return

    if not extra_paths:
        print("[*] 没有需要删除的文件")
        if missing_paths:
            print(f"[!] 但 CSV 中有 {len(missing_paths)} 个文件在当前目录缺失")
        return

    deleted_paths: list[str] = []
    failed_items: list[tuple[str, str]] = []

    print(f"[*] 开始删除 {len(extra_paths)} 个文件...")
    for index, relative_path in enumerate(extra_paths, 1):
        file_path = dataset_path / Path(relative_path)
        try:
            file_path.unlink()
            deleted_paths.append(relative_path)
            print(f"[{index}/{len(extra_paths)}] ✓ {relative_path}")
        except Exception as error:
            failed_items.append((relative_path, str(error)))
            print(f"[{index}/{len(extra_paths)}] ✗ {relative_path}: {error}")

    removed_dir_count = 0
    if delete_empty_dirs:
        removed_dir_count = remove_empty_directories(dataset_path)

    script_dir = csv_path.parent
    save_deleted_log(script_dir / "_dataset_sync_deleted.csv", deleted_paths)
    save_failed_log(script_dir / "_dataset_sync_failed.log", failed_items)

    print("")
    print(f"完成: 成功删除 {len(deleted_paths)}, 删除失败 {len(failed_items)}")
    if delete_empty_dirs:
        print(f"[*] 已删除空目录 {removed_dir_count} 个")
    if missing_paths:
        print(f"[!] CSV 中有 {len(missing_paths)} 个文件在当前目录缺失")


def run(
    mode: str,
    dataset_path: Path,
    csv_path: Path,
    only_image_files: bool,
    preview: bool,
    preview_limit: int,
    delete_empty_dirs: bool,
    exclude_paths: Iterable[Path] | None = None,
) -> None:
    """统一入口"""
    normalized_mode = mode.strip().lower()
    if normalized_mode == "collect":
        collect_manifest(
            dataset_path=dataset_path,
            csv_path=csv_path,
            only_image_files=only_image_files,
            preview=preview,
            preview_limit=preview_limit,
            exclude_paths=exclude_paths,
        )
        return

    if normalized_mode == "clean":
        clean_dataset(
            dataset_path=dataset_path,
            csv_path=csv_path,
            only_image_files=only_image_files,
            preview=preview,
            preview_limit=preview_limit,
            delete_empty_dirs=delete_empty_dirs,
            exclude_paths=exclude_paths,
        )
        return

    raise ValueError(f"不支持的模式: {mode}, 仅支持 collect 或 clean")


def main() -> None:
    # ============ 配置参数 (直接修改此处) ============
    MODE = "collect"                                # "collect" 或 "clean"
    DATASET_DIR = path_win2wsl(r"F:\dataset\select_clein_reblad_data_resized")  # 数据集根目录
    CSV_NAME = "dataset_manifest.csv"               # CSV 文件名, 保存在脚本同级目录
    ONLY_IMAGE_FILES = True                         # True 仅处理图片; False 处理所有文件
    PREVIEW = True                                  # 执行前显示预览并确认
    PREVIEW_LIMIT = 20                              # 预览显示条数
    DELETE_EMPTY_DIRS = True                        # clean 模式后是否清理空目录
    # ==============================================

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    dataset_path = Path(DATASET_DIR).expanduser()
    csv_path = script_dir / CSV_NAME

    exclude_paths = {script_path, csv_path}

    try:
        run(
            mode=MODE,
            dataset_path=dataset_path,
            csv_path=csv_path,
            only_image_files=ONLY_IMAGE_FILES,
            preview=PREVIEW,
            preview_limit=PREVIEW_LIMIT,
            delete_empty_dirs=DELETE_EMPTY_DIRS,
            exclude_paths=exclude_paths,
        )
    except Exception as error:
        print(f"[-] 执行失败: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
