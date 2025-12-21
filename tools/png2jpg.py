#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG转JPG工具
将指定路径下的PNG文件转换为JPG格式,支持递归处理子文件夹
"""

from pathlib import Path
from typing import Optional
from PIL import Image
import sys



def convert_png_to_jpg(png_path: Path, quality: int = 95) -> bool:
    """
    将PNG文件转换为JPG格式

    Args:
        png_path: PNG文件路径
        quality: JPG质量 (1-100)

    Returns:
        bool: 转换是否成功
    """
    try:
        # 打开PNG图片
        with Image.open(png_path) as img:
            # 如果图片有透明通道,转换为RGB
            if img.mode in ("RGBA", "LA", "P"):
                # 创建白色背景
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                rgb_img.paste(
                    img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
                )
                img = rgb_img
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # 生成JPG文件路径
            jpg_path = png_path.with_suffix(".jpg")

            # 保存为JPG
            img.save(jpg_path, "JPEG", quality=quality, optimize=True)

            print(f"✓ 转换成功: {png_path} -> {jpg_path}")
            return True

    except Exception as e:
        print(f"✗ 转换失败: {png_path} - {str(e)}")
        return False


def find_png_files(root_path: Path, filter_str: Optional[str] = None) -> list:
    """
    查找PNG文件

    Args:
        root_path: 根目录路径
        filter_str: 文件名过滤字符串

    Returns:
        list: PNG文件路径列表
    """
    png_files = []

    for file_path in root_path.rglob("*.png"):
        if filter_str is None or filter_str in file_path.name:
            png_files.append(file_path)

    return png_files


def main():
    # === 硬编码配置 ===
    path = r"E:\123pan\Downloads\1202_g_b"                   # 要处理的文件夹路径("."表示当前目录)
    filter_str = None            # 文件名过滤字符串(None表示不过滤)
    quality = 99                 # JPG质量设置 (1-100)
    delete_original = True      # 转换成功后删除原PNG文件
    dry_run = False              # 预览模式, 只显示将要转换的文件, 不执行转换

    # 检查路径是否存在
    root_path = Path(path)
    if not root_path.exists():
        print(f"错误: 路径不存在 - {root_path}")
        sys.exit(1)

    if not root_path.is_dir():
        print(f"错误: 路径不是文件夹 - {root_path}")
        sys.exit(1)

    # 查找PNG文件
    print(f"正在扫描文件夹: {root_path}")
    if filter_str:
        print(f"过滤条件: 文件名包含 '{filter_str}'")

    png_files = find_png_files(root_path, filter_str)

    if not png_files:
        print("未找到符合条件的PNG文件")
        return

    print(f"找到 {len(png_files)} 个PNG文件")

    # 预览模式
    if dry_run:
        print("\n预览模式 - 将要转换的文件:")
        for png_file in png_files:
            jpg_file = png_file.with_suffix(".jpg")
            print(f"  {png_file} -> {jpg_file}")
        return

    # 确认转换
    print("\n设置:")
    print(f"  JPG质量: {quality}")
    print(f"  删除原文件: {'是' if delete_original else '否'}")

    confirm = input(f"\n确定要转换这 {len(png_files)} 个文件吗? [y/N]: ").lower()
    if confirm not in ["y", "yes"]:
        print("已取消转换")
        return

    # 执行转换
    print("\n开始转换...")
    success_count = 0
    failed_count = 0

    for i, png_file in enumerate(png_files, 1):
        print(f"[{i}/{len(png_files)}] ", end="")

        if convert_png_to_jpg(png_file, quality):
            success_count += 1

            # 删除原文件
            if delete_original:
                try:
                    png_file.unlink()
                    print(f"  已删除原文件: {png_file}")
                except Exception as e:
                    print(f"  删除原文件失败: {png_file} - {str(e)}")
        else:
            failed_count += 1

    # 转换结果统计
    print("\n转换完成:")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {failed_count} 个文件")
    print(f"  总计: {len(png_files)} 个文件")


if __name__ == "__main__":
    main()
