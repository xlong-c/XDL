#!/usr/bin/env python3
"""
图片切割拼接工具 - 批量处理文件夹

功能:
1. 加载文件夹中的图片
2. 将图片按照等宽切成四张
3. 将第2,3,4张按照3,2,4的顺序重新拼接
4. 将分辨率设置到指定的scale

使用方法:
    修改下方 main() 中的配置参数后直接运行脚本
"""

import os
from pathlib import Path
from PIL import Image

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


def slice_image_equal_width(image, num_slices=4):
    width, height = image.size
    slice_width = width // num_slices

    slices = []
    for i in range(num_slices):
        left = i * slice_width
        right = (i + 1) * slice_width if i < num_slices - 1 else width
        slice_img = image.crop((left, 0, right, height))
        slices.append(slice_img)

    return slices


def reorder_and_concatenate(slices, order):
    reordered_slices = [slices[i - 1] for i in order]

    total_width = sum(s.width for s in reordered_slices)
    max_height = max(s.height for s in reordered_slices)

    result = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for slice_img in reordered_slices:
        result.paste(slice_img, (x_offset, 0))
        x_offset += slice_img.width

    return result


def _process_single(src: Path, dst: Path, scale: float, order: list):
    image = Image.open(src)

    if image.mode != "RGB":
        image = image.convert("RGB")

    slices = slice_image_equal_width(image, num_slices=4)
    reordered = reorder_and_concatenate(slices, order=order)

    if scale != 1.0:
        width, height = reordered.size
        new_size = (int(width * scale), int(height * scale))
        reordered = reordered.resize(new_size, Image.Resampling.LANCZOS)

    dst.parent.mkdir(parents=True, exist_ok=True)
    reordered.save(dst, quality=95)


def process_files(
    input_path: Path,
    output_path: Path,
    scale: float,
    order: list,
    keep_structure: bool = False,
    preview: bool = True,
    preview_limit: int = 5,
):
    files = list(find_images(input_path))
    if not files:
        print("未找到图片")
        return

    # 预览
    if preview:
        print("=" * 80)
        print(f"预览模式 (共 {len(files)} 个文件, 显示前 {preview_limit} 个):")
        print("=" * 80)
        for idx, (rel, src) in enumerate(files[:preview_limit], start=1):
            if keep_structure:
                dst = output_path / rel
            else:
                dst = output_path / f"{idx:06d}.jpg"
            print("\n[操作: 图片切割+拼接+缩放]")
            print(f"  源: {src}")
            print(f"  到: {dst}")
            print(f"  参数: scale={scale}, order={order}")
        if len(files) > preview_limit:
            print(f"\n... 还有 {len(files) - preview_limit} 个文件 ...")
        print("=" * 80)
        if input("\n确认执行? (y/n): ").lower() != "y":
            print("已取消")
            return

    # 处理
    ok = fail = 0
    output_path.mkdir(parents=True, exist_ok=True)
    for idx, (rel, src) in enumerate(files, start=1):
        try:
            if keep_structure:
                dst = output_path / rel
            else:
                dst = output_path / f"{idx:06d}.jpg"
            _process_single(src, dst, scale, order)
            print(f"✓ {src.name}")
            ok += 1
        except Exception as e:
            print(f"✗ {src.name}: {e}")
            fail += 1

    print(f"\n完成: 成功 {ok}, 失败 {fail}")


def main():
    # ============ 配置参数 ============
    INPUT = path_win2wsl(r"F:\dataset\select_clein")  # 输入文件夹
    OUTPUT = path_win2wsl(r"F:\dataset\select_clein_lq")  # 输出文件夹
    SCALE = 0.5  # 缩放比例 (0.0-1.0)
    ORDER = [3, 2, 4]  # 拼接顺序
    KEEP_STRUCTURE = False  # 保持原文件夹结构
    PREVIEW = True  # 预览模式
    PREVIEW_LIMIT = 5  # 预览条数
    # ==================================

    process_files(
        input_path=Path(INPUT),
        output_path=Path(OUTPUT),
        scale=SCALE,
        order=ORDER,
        keep_structure=KEEP_STRUCTURE,
        preview=PREVIEW,
        preview_limit=PREVIEW_LIMIT,
    )


if __name__ == "__main__":
    main()
