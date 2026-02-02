#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片格式转换与重命名工具

功能:
    批量将图片转换为 JPG 格式，并支持自定义命名规则

支持的源格式:
    JPG, JPEG, PNG, WEBP, BMP, TIFF, TIF

命名模式:
    1. 数字序号模式: 00001.jpg, 00002.jpg, ...
    2. 保留原文件名模式: 原文件名.jpg

特性:
    - 自动解决文件名冲突(自动添加 _1, _2 后缀)
    - JPG 源文件直接复制/移动(保持原质量)
    - 支持递归处理子目录
    - 支持预览模式(dry_run)
    - 可选处理完成后删除源文件

用法:
    修改 main() 中的配置参数，直接运行脚本
"""

import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 支持的源图片格式
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
# 视为已经是 JPG 的格式
JPG_EXTENSIONS = {".jpg", ".jpeg"}


def get_unique_path(
    directory: Path,
    filename: str,
    used_names: set,
    number_mode: bool = False,
    number: int = 0,
) -> Path:
    """
    在指定目录中获取唯一的文件路径。
    如果文件名已存在或在本次运行中已被占用，则添加后缀 _1, _2, ...
    如果启用 number_mode，则使用数字序号命名 (00001.jpg, 00002.jpg, ...)
    """
    suffix = ".jpg"  # 强制目标后缀为 .jpg

    if number_mode:
        # 数字序号模式: 00001.jpg, 00002.jpg, ...
        candidate_name = f"{number:05d}{suffix}"
        candidate = directory / candidate_name

        # 检查文件系统是否存在 OR 本次运行是否已占用
        if not candidate.exists() and candidate_name not in used_names:
            used_names.add(candidate_name)
            return candidate

        # 冲突解决（在数字序号模式下理论上不会发生，但保留此逻辑以防万一）
        counter = 1
        while True:
            candidate_name = f"{number:05d}_{counter}{suffix}"
            candidate = directory / candidate_name
            if not candidate.exists() and candidate_name not in used_names:
                used_names.add(candidate_name)
                return candidate
            counter += 1
    else:
        # 原始文件名模式
        name_stem = Path(filename).stem

        # 尝试原始名称
        candidate_name = f"{name_stem}{suffix}"
        candidate = directory / candidate_name

        # 检查文件系统是否存在 OR 本次运行是否已占用
        if not candidate.exists() and candidate_name not in used_names:
            used_names.add(candidate_name)
            return candidate

        # 冲突解决
        counter = 1
        while True:
            candidate_name = f"{name_stem}_{counter}{suffix}"
            candidate = directory / candidate_name
            if not candidate.exists() and candidate_name not in used_names:
                used_names.add(candidate_name)
                return candidate
            counter += 1


def convert_and_rename(
    source_dir: Path,
    target_dir: Path,
    delete_source: bool = False,
    recursive: bool = False,
    dry_run: bool = False,
    number_mode: bool = False,
    quality: int = 99,
):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        print(f"错误: 源路径不存在 - {source_dir}")
        return

    # 创建目标文件夹
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    # 收集图片文件
    pattern = "**/*" if recursive else "*"
    all_files = list(source_dir.glob(pattern))
    image_files = [
        f for f in all_files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    # 按文件名排序，确保处理顺序确定
    image_files.sort(key=lambda x: x.name)

    if not image_files:
        print(f"在 {source_dir} 中未找到支持的图片文件")
        return

    print(f"正在处理路径: {source_dir.absolute()}")
    print(f"目标路径: {target_dir.absolute()}")
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"递归模式: {'开启' if recursive else '关闭'}")
    print(f"删除源文件: {'是' if delete_source else '否'}")
    print(f"数字序号模式: {'开启' if number_mode else '关闭'}")
    print(f"模式: {'[预览]' if dry_run else '[实际执行]'}")
    print("-" * 40)

    success_count = 0
    fail_count = 0

    # 记录本次运行中目标文件夹已使用的文件名，防止冲突
    used_names = set()
    if target_dir.exists():
        for f in target_dir.glob("*.jpg"):
            used_names.add(f.name)

    # 使用 tqdm 显示进度
    for idx, src_path in enumerate(
        tqdm(image_files, desc="Processing", unit="img"), start=1
    ):
        try:
            # 确定目标路径 (解决冲突)
            dest_path = get_unique_path(
                target_dir,
                src_path.name,
                used_names,
                number_mode=number_mode,
                number=idx,
            )

            if dry_run:
                action = (
                    "Copy" if src_path.suffix.lower() in JPG_EXTENSIONS else "Convert"
                )
                print(f"[预览] {src_path.name} -> {dest_path.name} ({action})")
                success_count += 1
                continue

            # 如果已经是 JPG 格式，直接复制
            if src_path.suffix.lower() in JPG_EXTENSIONS:
                # 如果源和目标路径相同（原地处理且文件名没变），则跳过复制
                if src_path.resolve() != dest_path.resolve():
                    shutil.copy2(src_path, dest_path)
            else:
                # 否则执行格式转换
                with Image.open(src_path) as img:
                    # 转换为 RGB (处理 RGBA png 等)
                    rgb_img = img.convert("RGB")
                    # 保存为 JPG
                    rgb_img.save(dest_path, quality=quality)

            # 删除源文件 (如果启用且不是同一个文件)
            if delete_source:
                if src_path.resolve() != dest_path.resolve():
                    src_path.unlink()

            success_count += 1

        except Exception as e:
            print(f"\n✗ 处理失败: {src_path.name} - {str(e)}")
            fail_count += 1

    print(
        f"\n处理完成: {'预览' if dry_run else '成功'} {success_count}/{len(image_files)} 个文件, 失败 {fail_count} 个"
    )


def main():
    # === 配置区域 ===
    source_path = "/mnt/f/dataset/data_hq/clein_0128_sp2"  # 源图片文件夹路径
    target_path = "/mnt/f/dataset/data_hq/clein_0128_sp2_jpg"  # 目标文件夹路径
    delete_source = False  # 处理成功后是否删除源文件
    recursive = False  # 是否递归处理子文件夹
    dry_run = False  # 预览模式 (True: 只显示不修改, False: 实际执行)
    number_mode = False  # 数字序号模式 (True: 重命名为 00001.jpg, 00002.jpg 等, False: 保留原文件名)
    quality = 99
    # ================

    convert_and_rename(
        source_dir=source_path,
        target_dir=target_path,
        delete_source=delete_source,
        recursive=recursive,
        dry_run=dry_run,
        number_mode=number_mode,
        quality=quality,
    )


if __name__ == "__main__":
    main()
