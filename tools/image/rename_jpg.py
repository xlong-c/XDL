#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPG后缀大写转小写工具
将指定路径下的 .JPG 文件批量重命名为 .jpg
"""

import sys
from pathlib import Path


def rename_jpg_to_lowercase(
    root_path: Path, recursive: bool = False, dry_run: bool = False
):
    """
    将 .JPG 后缀修改为 .jpg
    """
    # 根据是否递归选择匹配模式
    pattern = "**/*.JPG" if recursive else "*.JPG"
    jpg_files = list(root_path.glob(pattern))

    if not jpg_files:
        print(f"在 {root_path} 中未找到任何 .JPG 文件")
        return

    print(f"找到 {len(jpg_files)} 个 .JPG 文件")
    if dry_run:
        print("--- 预览模式 (不执行实际重命名) ---")

    success_count = 0
    for file_path in jpg_files:
        # 构建新的路径，仅修改后缀
        new_path = file_path.with_suffix(".jpg")

        # 如果新旧路径完全相同（例如在不区分大小写的文件系统上，且匹配到了本身就是小写的情况，
        # 但 glob "*.JPG" 通常只匹配大写，除非系统本身不区分）
        if file_path == new_path:
            continue

        try:
            if dry_run:
                print(f"[预览] {file_path.relative_to(root_path)} -> {new_path.name}")
                success_count += 1
            else:
                # 执行重命名
                file_path.rename(new_path)
                print(f"✓ {file_path.name} -> {new_path.name}")
                success_count += 1
        except Exception as e:
            print(f"✗ 重命名失败: {file_path.name} - {str(e)}")

    print(
        f"\n处理完成: {'预览' if dry_run else '成功'} {success_count}/{len(jpg_files)} 个文件"
    )


def main():
    # === 配置区域 ===
    target_path = r"/root/autodl-tmp/100hairs/img_r"  # 目标文件夹路径
    recursive = False  # 是否递归处理子文件夹
    dry_run = True  # 预览模式 (True: 只显示不修改, False: 实际重命名)
    # ================

    root_path = Path(target_path)

    if not root_path.exists():
        print(f"错误: 路径不存在 - {root_path}")
        sys.exit(1)

    if not root_path.is_dir():
        print(f"错误: {root_path} 不是一个有效的文件夹")
        sys.exit(1)

    print(f"正在处理路径: {root_path.absolute()}")
    print(f"递归模式: {'开启' if recursive else '关闭'}")
    print(f"模式: {'[预览]' if dry_run else '[实际重命名]'}")
    print("-" * 40)

    rename_jpg_to_lowercase(root_path, recursive, dry_run)


if __name__ == "__main__":
    main()
