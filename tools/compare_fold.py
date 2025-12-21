# -*- coding: utf-8 -*-
import os
import csv
from pathlib import Path


def get_files_without_extension(folder_path, include_extension=False):
    """
    获取文件夹中的文件列表

    Args:
        folder_path: 文件夹路径
        include_extension: 是否包含文件扩展名

    Returns:
        文件名集合(去除或包含扩展名)
    """
    if not os.path.exists(folder_path):
        return set()

    files = set()
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            if include_extension:
                files.add(item)
            else:
                # 去除扩展名
                name, ext = os.path.splitext(item)
                files.add(name)

    return files


def compare_folders(folder_a, folder_b, output_csv, include_extension=False):
    """
    比较两个文件夹, 找出a有b没有的文件

    Args:
        folder_a: 文件夹A路径
        folder_b: 文件夹B路径
        output_csv: 输出CSV文件路径
        include_extension: 是否包含文件扩展名
    """
    # 获取两个文件夹的文件集合
    files_a = get_files_without_extension(folder_a, include_extension)
    files_b = get_files_without_extension(folder_b, include_extension)

    # 找出a有b没有的文件
    diff_files = files_a - files_b

    # 写入CSV文件
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['文件名'])

            for filename in sorted(diff_files):
                writer.writerow([filename])

        print(f'文件夹A中的文件数量: {len(files_a)}')
        print(f'文件夹B中的文件数量: {len(files_b)}')
        print(f'A有B没有的文件数量: {len(diff_files)}')
        print(f'结果已保存到: {output_csv}')

        # 显示部分结果
        if diff_files:
            print('\n前10个差异文件:')
            for i, filename in enumerate(sorted(diff_files)[:10]):
                print(f'  {i+1}. {filename}')

            if len(diff_files) > 10:
                print(f'  ... 还有 {len(diff_files) - 10} 个文件')

    except Exception as e:
        print(f'写入CSV文件时出错: {e}')


def main():
    # 直接在这里配置参数
    folder_a = r'E:\workspace\xdl\others\data\folder_a'  # 文件夹A路径
    folder_b = r'E:\workspace\xdl\others\data\folder_b'  # 文件夹B路径
    output_csv = r'E:\workspace\xdl\others\results\compare_result.csv'  # 输出CSV文件路径
    include_extension = False  # 是否包含文件扩展名(True/False)

    print('=' * 50)
    print('文件夹比较工具')
    print('=' * 50)
    print(f'文件夹A: {folder_a}')
    print(f'文件夹B: {folder_b}')
    print(f'输出文件: {output_csv}')
    print(f'包含扩展名: {"是" if include_extension else "否"}')
    print('=' * 50 + '\n')

    # 检查文件夹是否存在
    if not os.path.exists(folder_a):
        print(f'错误: 文件夹A不存在: {folder_a}')
        return

    if not os.path.exists(folder_b):
        print(f'错误: 文件夹B不存在: {folder_b}')
        return

    compare_folders(folder_a, folder_b, output_csv, include_extension)


if __name__ == '__main__':
    main()