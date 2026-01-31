#!/usr/bin/env python3
"""
批量重命名文件,按照序号命名,支持前缀
Usage: python rename_files.py
"""

import os
import sys
from pathlib import Path


def get_sorted_files(folder: str) -> list:
    """获取文件夹中所有文件并按文件名排序"""
    files = []
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            files.append(item)
    return sorted(files)


def rename_files(
    folder: str,
    prefix: str = "",
    start: int = 1,
    digits: int = 5,
    preview: bool = False,
):
    """
    将文件夹中的文件按序号重命名

    Args:
        folder: 目标文件夹路径
        prefix: 文件名前缀
        start: 起始序号
        digits: 序号位数(不足补零)
        preview: 是否仅预览,不实际重命名
    """
    if not os.path.exists(folder):
        print(f"错误: 路径不存在: {folder}")
        sys.exit(1)

    if not os.path.isdir(folder):
        print(f"错误: 不是文件夹: {folder}")
        sys.exit(1)

    files = get_sorted_files(folder)

    if not files:
        print(f"文件夹为空: {folder}")
        return

    mode_str = "[预览模式] 将要重命名" if preview else "重命名"
    print(f"\n{mode_str} {folder} 中的文件:\n")

    rename_plan = []
    for idx, filename in enumerate(files, start=start):
        old_path = os.path.join(folder, filename)
        ext = Path(filename).suffix
        new_name = (
            f"{prefix}_{idx:0{digits}d}{ext}" if prefix else f"{idx:0{digits}d}{ext}"
        )
        new_path = os.path.join(folder, new_name)
        rename_plan.append((old_path, new_path, filename, new_name))
        print(f"  {filename} -> {new_name}")

    print(f"\n共 {len(files)} 个文件")

    if preview:
        print("[预览完成]")
        return

    new_names = [new_name for _, _, _, new_name in rename_plan]
    if len(new_names) != len(set(new_names)):
        print("错误: 生成的文件名有冲突,请调整参数")
        sys.exit(1)

    confirm = input(f"\n确认重命名以上 {len(files)} 个文件? (y/n): ")
    if confirm.lower() != "y":
        print("已取消")
        return

    success_count = 0
    for old_path, new_path, old_name, new_name in rename_plan:
        try:
            os.rename(old_path, new_path)
            print(f"已重命名: {old_name} -> {new_name}")
            success_count += 1
        except Exception as e:
            print(f"重命名失败 {old_name}: {e}")

    print(f"\n完成! 共重命名 {success_count}/{len(files)} 个文件")


def main():
    folder = "/mnt" + "/f/dataset/data_hq/12yue_02_crop_highquanti"
    prefix = "1202"
    start_num = 1
    digits_num = 6
    preview_mode = False

    rename_files(
        folder, prefix=prefix, start=start_num, digits=digits_num, preview=preview_mode
    )


if __name__ == "__main__":
    main()
