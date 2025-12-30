#!/usr/bin/env python
"""
文件夹文件比对工具 - 比较两个文件夹的文件名差异

用法示例:
    # 比对两个文件夹
    python tools/dataset/compare_folders.py F:/dataset/10hair/bimg F:/dataset/10hair/zimg

    # 只比对特定扩展名的文件
    python compare_folders.py /path/to/folder1 /path/to/folder2 --ext png jpg

    # 递归比对子文件夹
    python compare_folders.py /path/to/folder1 /path/to/folder2 --recursive

    # 将差异保存到文件
    python compare_folders.py /path/to/folder1 /path/to/folder2 --output diff.txt
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def get_files(folder_path, extensions=None, recursive=False):
    """
    获取文件夹中的所有文件

    Args:
        folder_path: 文件夹路径
        extensions: 文件扩展名列表
        recursive: 是否递归获取子文件夹中的文件

    Returns:
        文件名集合
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        sys.exit(1)

    if not folder.is_dir():
        print(f"错误: '{folder_path}' 不是一个文件夹")
        sys.exit(1)

    files = set()

    if recursive:
        # 递归获取所有文件
        if extensions:
            for ext in extensions:
                files.update([f.name for f in folder.rglob(f"*.{ext}")])
                files.update([f.name for f in folder.rglob(f"*.{ext.upper()}")])
        else:
            files.update([f.name for f in folder.rglob("*") if f.is_file()])
    else:
        # 只获取顶层文件
        if extensions:
            for ext in extensions:
                files.update([f.name for f in folder.glob(f"*.{ext}")])
                files.update([f.name for f in folder.glob(f"*.{ext.upper()}")])
        else:
            files.update([f.name for f in folder.iterdir() if f.is_file()])

    return files


def compare_folders(folder1_path, folder2_path, extensions=None, recursive=False,
                    show_common=False, output_file=None):
    """
    比对两个文件夹的文件

    Args:
        folder1_path: 第一个文件夹路径
        folder2_path: 第二个文件夹路径
        extensions: 文件扩展名列表
        recursive: 是否递归比对子文件夹
        show_common: 是否显示共同的文件
        output_file: 输出文件路径
    """
    print("=" * 80)
    print("文件夹文件比对工具")
    print("=" * 80)
    print(f"文件夹1: {folder1_path}")
    print(f"文件夹2: {folder2_path}")
    print(f"扩展名过滤: {extensions if extensions else '全部'}")
    print(f"递归比对: {'是' if recursive else '否'}")
    print("-" * 80)

    # 获取两个文件夹的文件列表
    print("正在获取文件列表...")
    files1 = get_files(folder1_path, extensions, recursive)
    files2 = get_files(folder2_path, extensions, recursive)

    print(f"文件夹1: {len(files1)} 个文件")
    print(f"文件夹2: {len(files2)} 个文件")
    print("-" * 80)

    # 计算差异
    only_in_1 = files1 - files2  # 只在文件夹1中的文件
    only_in_2 = files2 - files1  # 只在文件夹2中的文件
    common = files1 & files2     # 共同的文件

    # 准备输出
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("文件夹文件比对报告")
    output_lines.append("=" * 80)
    output_lines.append(f"文件夹1: {folder1_path}")
    output_lines.append(f"文件夹2: {folder2_path}")
    output_lines.append(f"扩展名过滤: {extensions if extensions else '全部'}")
    output_lines.append(f"递归比对: {'是' if recursive else '否'}")
    output_lines.append("-" * 80)
    output_lines.append(f"文件夹1: {len(files1)} 个文件")
    output_lines.append(f"文件夹2: {len(files2)} 个文件")
    output_lines.append("-" * 80)

    # 显示结果
    if only_in_1:
        print(f"\n✓ 只在文件夹1中 ({len(only_in_1)} 个文件):")
        output_lines.append(f"\n只在文件夹1中 ({len(only_in_1)} 个文件):")
        for filename in sorted(only_in_1):
            print(f"  - {filename}")
            output_lines.append(f"  - {filename}")
    else:
        print(f"\n✓ 文件夹1中没有独有文件")
        output_lines.append("\n文件夹1中没有独有文件")

    if only_in_2:
        print(f"\n✓ 只在文件夹2中 ({len(only_in_2)} 个文件):")
        output_lines.append(f"\n只在文件夹2中 ({len(only_in_2)} 个文件):")
        for filename in sorted(only_in_2):
            print(f"  - {filename}")
            output_lines.append(f"  - {filename}")
    else:
        print(f"\n✓ 文件夹2中没有独有文件")
        output_lines.append("\n文件夹2中没有独有文件")

    if show_common and common:
        print(f"\n✓ 共同的文件 ({len(common)} 个文件):")
        output_lines.append(f"\n共同的文件 ({len(common)} 个文件):")
        for filename in sorted(common):
            print(f"  - {filename}")
            output_lines.append(f"  - {filename}")
    elif show_common:
        print(f"\n✓ 没有共同的文件")
        output_lines.append("\n没有共同的文件")

    print("\n" + "=" * 80)
    print(f"统计: 文件夹1独有={len(only_in_1)}, 文件夹2独有={len(only_in_2)}, 共同={len(common)}")
    print("=" * 80)

    output_lines.append("\n" + "=" * 80)
    output_lines.append(
        f"统计: 文件夹1独有={len(only_in_1)}, 文件夹2独有={len(only_in_2)}, 共同={len(common)}"
    )
    output_lines.append("=" * 80)

    # 保存到文件
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"\n保存文件时出错: {str(e)}")

    return only_in_1, only_in_2, common


def main():
    parser = argparse.ArgumentParser(
        description="比较两个文件夹的文件名差异",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本比对
  %(prog)s /path/to/folder1 /path/to/folder2

  # 只比对PNG和JPG文件
  %(prog)s /path/to/folder1 /path/to/folder2 --ext png jpg

  # 递归比对子文件夹
  %(prog)s /path/to/folder1 /path/to/folder2 --recursive

  # 显示共同的文件
  %(prog)s /path/to/folder1 /path/to/folder2 --show-common

  # 保存结果到文件
  %(prog)s /path/to/folder1 /path/to/folder2 --output diff.txt

  # 组合使用
  %(prog)s /path/to/folder1 /path/to/folder2 --ext png --recursive --show-common --output diff.txt
        """
    )

    parser.add_argument(
        "folder1",
        help="第一个文件夹路径"
    )

    parser.add_argument(
        "folder2",
        help="第二个文件夹路径"
    )

    parser.add_argument(
        "--ext",
        nargs="+",
        help="文件扩展名过滤，例如: --ext png jpg"
    )

    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="递归比对子文件夹"
    )

    parser.add_argument(
        "--show-common", "-c",
        action="store_true",
        help="显示共同的文件"
    )

    parser.add_argument(
        "--output", "-o",
        help="将结果保存到文件"
    )

    args = parser.parse_args()

    compare_folders(
        args.folder1,
        args.folder2,
        extensions=args.ext,
        recursive=args.recursive,
        show_common=args.show_common,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
