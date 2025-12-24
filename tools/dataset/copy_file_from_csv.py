#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV文件复制脚本
读取CSV中记录的文件路径, 将这些文件复制到指定的目标文件夹c中
"""

import os
import csv
import shutil
from typing import List, Tuple


def read_csv_files(csv_path: str) -> List[str]:
    """
    从CSV文件中读取要复制的文件路径列表

    Args:
        csv_path: CSV文件路径

    Returns:
        要复制的文件路径列表
    """
    files_to_copy = []

    if not os.path.exists(csv_path):
        print(f"错误: CSV文件 {csv_path} 不存在")
        return files_to_copy

    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            # 查找文件路径相关的列
            file_path_column = None
            possible_columns = ['文件路径', '实际文件名', '路径', 'path', 'file_path', 'filepath']

            if reader.fieldnames:
                for col in possible_columns:
                    if col in reader.fieldnames:
                        file_path_column = col
                        break

            if file_path_column is None:
                # 如果没有找到明确的路径列, 使用最后一列
                file_path_column = reader.fieldnames[-1] if reader.fieldnames else None
                print(f"警告: 未找到明确的文件路径列, 使用最后一列: {file_path_column}")

            for row in reader:
                if file_path_column and file_path_column in row:
                    file_path = row[file_path_column].strip()
                    if file_path and file_path != '未找到对应文件':
                        files_to_copy.append(file_path)

        print(f"从CSV中读取了 {len(files_to_copy)} 个文件路径")

    except Exception as e:
        print(f"读取CSV文件时出错: {e}")

    return files_to_copy


def copy_files_to_target(file_paths: List[str], target_folder: str) -> Tuple[int, int, List[str]]:
    """
    将文件列表中的文件复制到目标文件夹

    Args:
        file_paths: 要复制的文件路径列表
        target_folder: 目标文件夹路径

    Returns:
        (成功复制数量, 失败数量, 失败的文件列表)
    """
    success_count = 0
    fail_count = 0
    failed_files = []

    # 创建目标文件夹(如果不存在)
    os.makedirs(target_folder, exist_ok=True)
    print(f"目标文件夹: {target_folder}")

    for file_path in file_paths:
        if not file_path:
            continue

        try:
            if os.path.exists(file_path):
                # 获取文件名
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_folder, filename)

                # 如果目标文件已存在, 添加序号
                counter = 1
                original_target = target_path
                while os.path.exists(target_path):
                    name, ext = os.path.splitext(original_target)
                    target_path = f"{name}_{counter}{ext}"
                    counter += 1

                # 复制文件
                shutil.copy2(file_path, target_path)
                success_count += 1
                print(f"✓ 复制: {filename} -> {target_folder}")
            else:
                fail_count += 1
                failed_files.append(file_path)
                print(f"✗ 文件不存在: {file_path}")

        except Exception as e:
            fail_count += 1
            failed_files.append(file_path)
            print(f"✗ 复制失败 {file_path}: {e}")

    return success_count, fail_count, failed_files


def save_failed_log(failed_files: List[str], target_folder: str):
    """
    保存失败的文件列表到日志文件

    Args:
        failed_files: 失败的文件路径列表
        target_folder: 目标文件夹路径
    """
    if not failed_files:
        return

    log_path = os.path.join(target_folder, "copy_failed_log.txt")
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("复制失败的文件列表:\n")
            f.write("=" * 50 + "\n")
            for i, file_path in enumerate(failed_files, 1):
                f.write(f"{i}. {file_path}\n")
        print(f"失败日志已保存到: {log_path}")
    except Exception as e:
        print(f"保存失败日志时出错: {e}")


def main():
    # 直接在代码中指定参数
    csv_path = "temp.csv"  # CSV文件路径
    target_folder = r"F:\1124_none_g"  # 目标文件夹路径

    print("=" * 60)
    print("CSV文件复制脚本")
    print("=" * 60)
    print(f"CSV文件: {csv_path}")
    print(f"目标文件夹: {target_folder}")
    print("=" * 60)

    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件 {csv_path} 不存在")
        print("请确保CSV文件路径正确, 或者先运行image_count.py生成CSV文件")
        return

    # 读取CSV中的文件路径
    files_to_copy = read_csv_files(csv_path)

    if not files_to_copy:
        print("没有找到要复制的文件")
        return

    print(f"\n准备复制 {len(files_to_copy)} 个文件...")
    print("开始复制文件...")

    # 复制文件
    success_count, fail_count, failed_files = copy_files_to_target(files_to_copy, target_folder)

    # 显示结果
    print("\n" + "=" * 60)
    print("复制完成!")
    print("=" * 60)
    print(f"成功复制: {success_count} 个文件")
    print(f"复制失败: {fail_count} 个文件")
    print(f"成功率: {(success_count / len(files_to_copy) * 100):.2f}%")

    if failed_files:
        print(f"\n失败的文件列表 (前10个):")
        for i, file_path in enumerate(failed_files[:10], 1):
            print(f"{i:2d}. {file_path}")

        if len(failed_files) > 10:
            print(f"... 还有 {len(failed_files) - 10} 个文件")

        # 保存失败日志
        save_failed_log(failed_files, target_folder)

    print(f"\n所有文件已尝试复制到: {target_folder}")


if __name__ == "__main__":
    main()