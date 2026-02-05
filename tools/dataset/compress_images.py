#!/usr/bin/env python3
"""
图像压缩工具 - 按指定比例压缩图片

功能:
- 将输入文件夹中的所有图片压缩到指定比例
- 支持多种图片格式 (jpg, png, bmp, gif, tiff, webp)
- 输出为 JPEG 格式，质量分数 100
- 保持原文件夹结构

Usage: python compress_images.py
"""

import os
import sys
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def path_win2wsl(win_path: str) -> str:
    if not win_path:
        return win_path
    win_path = win_path.strip()
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path


def find_images(input_path: Path):
    for root, _, files in os.walk(input_path):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p.relative_to(input_path), p


def compress_image(src: Path, dst: Path, scale: float, quality: int = 100):
    img = Image.open(src)
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    if new_width < 1 or new_height < 1:
        raise ValueError(f"压缩比例过小，输出尺寸无效: {new_width}x{new_height}")
    
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_resized.save(dst, quality=quality, optimize=True)


def _process_single(args, output_path: Path, scale: float, quality: int):
    rel, src = args
    filename = src.name
    
    try:
        dst = output_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        compress_image(src, dst, scale, quality)
        
        return (filename, True, None, f"{src.stat().st_size} -> {dst.stat().st_size}")
        
    except Exception as e:
        return (filename, False, str(e), None)


def process_files(
    input_path: Path,
    output_path: Path,
    scale: float = 0.5,
    quality: int = 100,
    preview: bool = True,
    preview_limit: int = 5,
    workers: int = None,
):
    if not input_path.exists():
        print(f"错误: 路径不存在: {input_path}")
        sys.exit(1)
    
    files = list(find_images(input_path))
    if not files:
        print("未找到图片")
        return
    
    print("=" * 80)
    print(f"输入文件夹: {input_path}")
    print(f"输出文件夹: {output_path}")
    print("=" * 80)
    print(f"\n找到 {len(files)} 个文件待处理")
    print(f"压缩比例: {scale} (原始尺寸的 {scale*100:.1f}%)")
    print(f"质量分数: {quality}")
    
    if preview:
        print("\n" + "=" * 80)
        print(f"预览模式 (共 {len(files)} 个文件, 显示前 {preview_limit} 个):")
        print("=" * 80)
        
        for rel, src in files[:preview_limit]:
            dst = output_path / rel
            img = Image.open(src)
            width, height = img.size
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            print("\n[操作: 图像压缩]")
            print(f"  源: {src}")
            print(f"  尺寸: {width}x{height}")
            print(f"  到: {dst}")
            print(f"  新尺寸: {new_width}x{new_height}")
            print(f"  参数: scale={scale}, quality={quality}")
        
        if len(files) > preview_limit:
            print(f"\n... 还有 {len(files) - preview_limit} 个文件 ...")
        
        print("=" * 80)
        if input("\n确认执行? (y/n): ").lower() != "y":
            print("已取消")
            return
    
    worker_count = workers if workers else cpu_count()
    print(f"\n使用 {worker_count} 个进程处理 {len(files)} 个文件...")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    worker_func = partial(
        _process_single,
        output_path=output_path,
        scale=scale,
        quality=quality,
    )
    
    ok = fail = 0
    failed_files = []
    total_size_before = 0
    total_size_after = 0
    
    with Pool(processes=worker_count) as pool:
        results = pool.map(worker_func, files)
    
    for name, success, error, size_info in results:
        if success and size_info:
            size_before, size_after = size_info.split(" -> ")
            size_before = int(size_before)
            size_after = int(size_after)
            total_size_before += size_before
            total_size_after += size_after
            
            compression_ratio = size_after / size_before if size_before > 0 else 0
            print(f"✓ {name} ({size_before:,} → {size_after:,} bytes, {compression_ratio:.1%})")
            ok += 1
        else:
            print(f"✗ {name}: {error}")
            fail += 1
            failed_files.append((name, error))
    
    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    
    if total_size_before > 0:
        compression_ratio = total_size_after / total_size_before
        print("\n压缩统计:")
        print(f"  原始总大小: {total_size_before:,} bytes ({total_size_before/1024/1024:.2f} MB)")
        print(f"  压缩后大小: {total_size_after:,} bytes ({total_size_after/1024/1024:.2f} MB)")
        print(f"  压缩比例: {compression_ratio:.1%}")
        print(f"  节省空间: {total_size_before - total_size_after:,} bytes ({(total_size_before - total_size_after)/1024/1024:.2f} MB)")
    
    if failed_files:
        log_path = output_path / "_compress_failed.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# 压缩失败记录\n")
            f.write(f"# 总计: {fail} 个文件失败\n\n")
            for name, error in failed_files:
                f.write(f"{name}: {error}\n")
        print(f"\n失败文件列表已保存到: {log_path}")


def main():
    INPUT = path_win2wsl(r"F:\dataset\select_3w\all")
    OUTPUT = path_win2wsl(r"F:\dataset\select_3w\all_lq")
    SCALE = 0.5
    QUALITY = 100
    PREVIEW = True
    PREVIEW_LIMIT = 5
    WORKERS = None
    
    process_files(
        input_path=Path(INPUT),
        output_path=Path(OUTPUT),
        scale=SCALE,
        quality=QUALITY,
        preview=PREVIEW,
        preview_limit=PREVIEW_LIMIT,
        workers=WORKERS,
    )


if __name__ == "__main__":
    main()
