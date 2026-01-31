#!/usr/bin/env python3
"""
将文件夹b中与文件夹a重复的文件复制或移动到文件夹c
Usage: python copy_duplicates.py
"""

import os
import sys
import shutil


def get_all_files(folder: str) -> set:
    """获取文件夹中所有文件的文件名集合"""
    files = set()
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            files.add(item)
    return files


def process_duplicate_files(
    folder_a: str, folder_b: str, folder_c: str, mode: str = "copy", preview: bool = False
):
    """
    将 folder_b 中与 folder_a 重复的文件复制或移动到 folder_c
    
    Args:
        folder_a: 参考文件夹路径（对比用）
        folder_b: 源文件夹路径（从中处理重复文件）
        folder_c: 目标文件夹路径（处理到此处）
        mode: "copy" 或 "move"
        preview: 是否仅预览，不实际操作
    """
    for folder in [folder_a, folder_b]:
        if not os.path.exists(folder):
            print(f"错误: 路径不存在: {folder}")
            sys.exit(1)
    
    if not preview and not os.path.exists(folder_c):
        os.makedirs(folder_c)
        print(f"创建目录: {folder_c}")
    
    files_a = get_all_files(folder_a)
    files_b = get_all_files(folder_b)
    
    duplicates = files_a & files_b
    
    action_str = "复制" if mode == "copy" else "移动"
    mode_str = f"[预览模式] 将要{action_str}" if preview else action_str
    
    print(f"\n{mode_str} {folder_b} 中与 {folder_a} 重复的文件到 {folder_c}:\n")
    
    if not duplicates:
        print("(没有重复文件)")
        return
    
    for filename in sorted(duplicates):
        src = os.path.join(folder_b, filename)
        dst = os.path.join(folder_c, filename)
        print(f"  {src}")
        print(f"    -> {dst}\n")
    
    print(f"共 {len(duplicates)} 个重复文件")
    
    if preview:
        print("[预览完成]")
        return
    
    confirm = input(f"\n确认{action_str}以上 {len(duplicates)} 个文件? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        return
    
    processed_count = 0
    for filename in duplicates:
        src_path = os.path.join(folder_b, filename)
        dst_path = os.path.join(folder_c, filename)
        try:
            if mode == "copy":
                shutil.copy2(src_path, dst_path)
                print(f"已复制: {filename}")
            else:
                shutil.move(src_path, dst_path)
                print(f"已移动: {filename}")
            processed_count += 1
        except Exception as e:
            print(f"{action_str}失败 {filename}: {e}")
    
    print(f"\n完成! 共{action_str} {processed_count} 个文件到 {folder_c}")


def main():
    print("将文件夹b中与文件夹a重复的文件复制或移动到文件夹c")
    folder_a = "/mnt/f/crop_1124"
    folder_b = "/mnt/f/raw_pics/good/合集"
    folder_c = "/mnt/f/raw_pics/good/1124"
    # 模式选择 "copy" / "move"
    mode = "move"
    # 预览模式
    preview_mode = False
    
    process_duplicate_files(folder_a, folder_b, folder_c, mode=mode, preview=preview_mode)


if __name__ == "__main__":
    main()
