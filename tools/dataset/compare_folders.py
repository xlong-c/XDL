#!/usr/bin/env python
"""
文件夹文件比对工具 - 比较两个文件夹的文件名差异

用法示例:
    直接修改 main() 函数中的参数,然后运行:
    python tools/dataset/compare_folders.py
"""

import shutil
import sys
from pathlib import Path


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
        if extensions:
            for ext in extensions:
                files.update([f.name for f in folder.rglob(f"*.{ext}")])
                files.update([f.name for f in folder.rglob(f"*.{ext.upper()}")])
        else:
            files.update([f.name for f in folder.rglob("*") if f.is_file()])
    else:
        if extensions:
            for ext in extensions:
                files.update([f.name for f in folder.glob(f"*.{ext}")])
                files.update([f.name for f in folder.glob(f"*.{ext.upper()}")])
        else:
            files.update([f.name for f in folder.iterdir() if f.is_file()])

    return files


def copy_symmetric_diff(folder1_path, folder2_path, folder3_path, only_in_1, only_in_2):
    """
    将两个文件夹的对称差集文件复制到第三个文件夹

    Args:
        folder1_path: 第一个文件夹路径
        folder2_path: 第二个文件夹路径
        folder3_path: 目标文件夹路径
        only_in_1: 只在文件夹1中的文件名集合
        only_in_2: 只在文件夹2中的文件名集合
    """
    folder1 = Path(folder1_path)
    folder2 = Path(folder2_path)
    folder3 = Path(folder3_path)

    # 创建目标文件夹
    folder3.mkdir(parents=True, exist_ok=True)

    # 创建子文件夹用于区分来源
    folder3_only1 = folder3 / "only_in_folder1"
    folder3_only2 = folder3 / "only_in_folder2"
    folder3_only1.mkdir(exist_ok=True)
    folder3_only2.mkdir(exist_ok=True)

    copied_count = 0
    failed_count = 0

    # 复制文件夹1独有的文件
    for filename in only_in_1:
        src = folder1 / filename
        dst = folder3_only1 / filename
        try:
            shutil.copy2(src, dst)
            copied_count += 1
        except Exception as e:
            print(f"  复制失败 {filename}: {e}")
            failed_count += 1

    # 复制文件夹2独有的文件
    for filename in only_in_2:
        src = folder2 / filename
        dst = folder3_only2 / filename
        try:
            shutil.copy2(src, dst)
            copied_count += 1
        except Exception as e:
            print(f"  复制失败 {filename}: {e}")
            failed_count += 1

    print(f"\n✓ 复制完成: 成功 {copied_count} 个, 失败 {failed_count} 个")
    print(f"  目标文件夹: {folder3}")
    print(f"    - {folder3_only1.name}: {len(only_in_1)} 个文件")
    print(f"    - {folder3_only2.name}: {len(only_in_2)} 个文件")


def compare_folders(
    folder1_path,
    folder2_path,
    extensions=None,
    recursive=False,
    output_file=None,
    folder3_path=None,
):
    """
    比对两个文件夹的文件

    Args:
        folder1_path: 第一个文件夹路径
        folder2_path: 第二个文件夹路径
        extensions: 文件扩展名列表
        recursive: 是否递归比对子文件夹
        output_file: 输出文件路径
        folder3_path: 对称差集复制目标文件夹,None表示不复制
    """
    print("=" * 80)
    print("文件夹文件比对工具")
    print("=" * 80)
    print(f"文件夹1: {folder1_path}")
    print(f"文件夹2: {folder2_path}")
    print(f"扩展名过滤: {extensions if extensions else '全部'}")
    print(f"递归比对: {'是' if recursive else '否'}")
    print("-" * 80)

    print("正在获取文件列表...")
    files1 = get_files(folder1_path, extensions, recursive)
    files2 = get_files(folder2_path, extensions, recursive)

    print(f"文件夹1: {len(files1)} 个文件")
    print(f"文件夹2: {len(files2)} 个文件")
    print("-" * 80)

    only_in_1 = files1 - files2
    only_in_2 = files2 - files1
    common = files1 & files2

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

    if only_in_1:
        print(f"\n✓ 只在文件夹1中 ({len(only_in_1)} 个文件):")
        output_lines.append(f"\n只在文件夹1中 ({len(only_in_1)} 个文件):")
        for i, filename in enumerate(sorted(only_in_1)):
            if i >= 30:
                print(f"  ... 还有 {len(only_in_1) - 30} 个文件未显示")
                output_lines.append(f"  ... 还有 {len(only_in_1) - 30} 个文件未显示")
                break
            print(f"  - {filename}")
            output_lines.append(f"  - {filename}")
    else:
        print("\n✓ 文件夹1中没有独有文件")
        output_lines.append("\n文件夹1中没有独有文件")

    if only_in_2:
        print(f"\n✓ 只在文件夹2中 ({len(only_in_2)} 个文件):")
        output_lines.append(f"\n只在文件夹2中 ({len(only_in_2)} 个文件):")
        for i, filename in enumerate(sorted(only_in_2)):
            if i >= 30:
                print(f"  ... 还有 {len(only_in_2) - 30} 个文件未显示")
                output_lines.append(f"  ... 还有 {len(only_in_2) - 30} 个文件未显示")
                break
            print(f"  - {filename}")
            output_lines.append(f"  - {filename}")
    else:
        print("\n✓ 文件夹2中没有独有文件")
        output_lines.append("\n文件夹2中没有独有文件")

    if common:
        print(f"\n✓ 共同的文件 ({len(common)} 个文件):")
        output_lines.append(f"\n共同的文件 ({len(common)} 个文件):")
        for i, filename in enumerate(sorted(common)):
            if i >= 30:
                print(f"  ... 还有 {len(common) - 30} 个文件未显示")
                output_lines.append(f"  ... 还有 {len(common) - 30} 个文件未显示")
                break
            print(f"  - {filename}")
            output_lines.append(f"  - {filename}")
    else:
        print("\n✓ 没有共同的文件")
        output_lines.append("\n没有共同的文件")

    print("\n" + "=" * 80)
    print(
        f"统计: 文件夹1独有={len(only_in_1)}, 文件夹2独有={len(only_in_2)}, 共同={len(common)}"
    )
    print("=" * 80)

    output_lines.append("\n" + "=" * 80)
    output_lines.append(
        f"统计: 文件夹1独有={len(only_in_1)}, 文件夹2独有={len(only_in_2)}, 共同={len(common)}"
    )
    output_lines.append("=" * 80)

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"\n保存文件时出错: {str(e)}")

    # 复制对称差集到folder3
    if folder3_path and (only_in_1 or only_in_2):
        print("\n" + "-" * 80)
        print("正在复制对称差集文件...")
        copy_symmetric_diff(folder1_path, folder2_path, folder3_path, only_in_1, only_in_2)

    return only_in_1, only_in_2, common


def main():
    """
    主函数 - 在此配置参数

    配置项:
        folder1: 第一个文件夹路径
        folder2: 第二个文件夹路径
        folder3: 对称差集复制目标文件夹,None表示不复制
        extensions: 文件扩展名列表,例如 ["png", "jpg"],设为 None 表示全部文件
        recursive: 是否递归比对子文件夹,True/False
        output_file: 输出文件路径,设为 None 表示不保存到文件
    """
    # ========================================
    # 配置参数区域 - 根据需要修改以下参数
    # ========================================

    # 文件夹路径
    folder1 = "/mnt/f/raw_pics/good/1203"  # 第一个文件夹路径
    folder2 = "/mnt/f/raw_pics/good/combine"  # 第二个文件夹路径
    folder3 = None  # 对称差集复制目标,设为 None 表示不复制

    # 文件扩展名过滤,例如: ["png", "jpg", "jpeg"]
    extensions = None

    # 是否递归比对子文件夹
    recursive = False

    # 输出文件路径,设为 None 表示不保存到文件
    output_file = None

    # ========================================
    # 执行比对
    # ========================================

    compare_folders(
        folder1,
        folder2,
        extensions=extensions,
        recursive=recursive,
        output_file=output_file,
        folder3_path=folder3,
    )


if __name__ == "__main__":
    main()
