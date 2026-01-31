#!/usr/bin/env python3
"""
删除文件夹a中与文件夹b重复的文件
Usage: python delete_files.py
"""

import os
import sys


def get_all_files(folder: str) -> set:
    """获取文件夹中所有文件的文件名集合（不包括路径）"""
    files = set()
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            files.add(item)
    return files


def delete_duplicate_files(folder_a: str, folder_b: str, preview: bool = False):
    """
    删除 folder_a 中与 folder_b 文件名重复的文件
    
    Args:
        folder_a: 要清理的文件夹路径
        folder_b: 参考文件夹路径（对比用）
        preview: 是否仅预览，不实际删除
    """
    if not os.path.exists(folder_a):
        print(f"错误: 路径不存在: {folder_a}")
        sys.exit(1)
    
    if not os.path.exists(folder_b):
        print(f"错误: 路径不存在: {folder_b}")
        sys.exit(1)
    
    files_a = get_all_files(folder_a)
    files_b = get_all_files(folder_b)
    
    duplicates = files_a & files_b
    
    mode_str = "[预览模式] 将要删除" if preview else "删除"
    print(f"\n{mode_str} {folder_a} 中与 {folder_b} 重复的文件:\n")
    
    if not duplicates:
        print("(没有重复文件)")
        return
    
    for filename in sorted(duplicates):
        print(f"  {filename}")
    
    print(f"\n共 {len(duplicates)} 个重复文件")
    
    if preview:
        print("[预览完成]")
        return
    
    confirm = input(f"\n确认删除以上 {len(duplicates)} 个文件? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        return
    
    deleted_count = 0
    for filename in duplicates:
        file_path = os.path.join(folder_a, filename)
        try:
            os.remove(file_path)
            print(f"已删除: {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"删除失败 {filename}: {e}")
    
    print(f"\n完成! 共删除 {deleted_count} 个文件")


def main():
    folder_a = "/mnt/f/raw_pics/good/合集"
    folder_b = "/mnt/f/raw_pics/good/1124"
    preview_mode = True
    
    delete_duplicate_files(folder_a, folder_b, preview=preview_mode)


if __name__ == "__main__":
    main()
