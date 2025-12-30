#!/usr/bin/env python
"""
文件重命名工具 - 去除下划线连接的文件名中的指定部分

用法示例:
    # 去掉第1和第4部分，保留第2、3部分（索引从0开始）
    python rename_files.py /path/to/folder --keep 1 2

    # 预览模式（不实际重命名，只显示会做什么）
    python rename_files.py /path/to/folder --keep 1 2 --preview

    # 指定文件扩展名
    python rename_files.py /path/to/folder --keep 1 2 --ext png
"""

import os
import sys
import argparse
from pathlib import Path


def rename_files(folder_path, keep_indices, extensions=None, preview=False, dry_run=False):
    """
    重命名文件夹中的文件

    Args:
        folder_path: 文件夹路径
        keep_indices: 要保留的部分索引列表（从0开始）
        extensions: 文件扩展名列表，如 ['png', 'jpg']
        preview: 是否只预览不执行
        dry_run: 同preview
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    if not folder.is_dir():
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return

    # 获取所有文件
    if extensions:
        files = []
        for ext in extensions:
            files.extend(folder.glob(f"*.{ext}"))
            files.extend(folder.glob(f"*.{ext.upper()}"))
    else:
        files = [f for f in folder.iterdir() if f.is_file()]

    if not files:
        print(f"在 '{folder_path}' 中没有找到文件")
        return

    print(f"找到 {len(files)} 个文件")
    print(f"保留索引: {keep_indices}")
    print(f"扩展名过滤: {extensions if extensions else '全部'}")
    print("-" * 80)

    success_count = 0
    skip_count = 0
    error_count = 0

    for file_path in files:
        try:
            # 分离文件名和扩展名
            stem = file_path.stem  # 不带扩展名的文件名
            ext = file_path.suffix  # 扩展名（包含点）

            # 用下划线分割
            parts = stem.split('_')

            # 检查是否有足够的部分
            if len(parts) < max(keep_indices) + 1:
                print(
                    f"跳过: {file_path.name} (部分数不足: {len(parts)} < {max(keep_indices) + 1})")
                skip_count += 1
                continue

            # 保留指定的部分
            new_parts = [parts[i] for i in keep_indices if i < len(parts)]
            new_name = '_'.join(new_parts) + ext

            # 检查新文件名是否与旧文件名相同
            if new_name == file_path.name:
                print(f"跳过: {file_path.name} (文件名未改变)")
                skip_count += 1
                continue

            # 构建新路径
            new_path = file_path.parent / new_name

            # 检查目标文件是否已存在
            if new_path.exists() and new_path != file_path:
                print(f"警告: {new_name} 已存在，跳过 {file_path.name}")
                skip_count += 1
                continue

            # 显示重命名信息
            if preview or dry_run:
                print(f"重命名: {file_path.name} -> {new_name}")
            else:
                # 实际重命名
                file_path.rename(new_path)
                print(f"✓ {file_path.name} -> {new_name}")

            success_count += 1

        except Exception as e:
            print(f"✗ 错误处理 {file_path.name}: {str(e)}")
            error_count += 1

    print("-" * 80)
    print(f"完成! 成功: {success_count}, 跳过: {skip_count}, 错误: {error_count}")

    if preview or dry_run:
        print("[预览模式] 使用 --no-preview 来实际执行重命名")

#  python tools/dataset/rename_files.py F:\dataset\10hair\bimg --keep 1 2 --preview
def main():
    parser = argparse.ArgumentParser(
        description="重命名文件 - 去除下划线连接的文件名中的指定部分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 去掉第1和第4部分，保留第2、3部分（索引从0开始）
  %(prog)s /path/to/folder --keep 1 2

  # 预览模式（不实际重命名）
  %(prog)s /path/to/folder --keep 1 2 --preview

  # 只处理PNG文件
  %(prog)s /path/to/folder --keep 1 2 --ext png

  # 处理多种文件类型
  %(prog)s /path/to/folder --keep 1 2 --ext png jpg jpeg

注意: 索引从0开始，所以 --keep 1 2 表示保留第2和第3部分
        """
    )

    parser.add_argument(
        "folder",
        help="要处理的文件夹路径"
    )

    parser.add_argument(
        "--keep",
        type=int,
        nargs="+",
        required=True,
        help="要保留的部分索引（从0开始），例如: --keep 1 2 表示保留第2和第3部分"
    )

    parser.add_argument(
        "--ext",
        nargs="+",
        help="文件扩展名过滤，例如: --ext png jpg"
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="预览模式，不实际重命名文件"
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="实际执行重命名（取消预览模式）"
    )

    args = parser.parse_args()

    # 默认使用预览模式，除非指定 --no-preview
    preview = not args.no_preview

    rename_files(
        args.folder,
        args.keep,
        extensions=args.ext,
        preview=preview
    )


if __name__ == "__main__":
    main()
