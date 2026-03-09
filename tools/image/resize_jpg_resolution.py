#!/usr/bin/env python3
"""
JPG 分辨率批量调整工具

功能:
- 递归扫描输入目录中的 .jpg / .jpeg 文件
- 将图片缩放到指定分辨率
- 默认保持原目录结构输出到新目录
- 尽量继承原 JPEG 的量化表、subsampling、EXIF、ICC profile
- 支持预览确认、多进程处理、失败日志记录
"""

from __future__ import annotations

import os
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageFile, ImageOps, JpegImagePlugin


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
ORIENTATION_TAG = 274


def path_win2wsl(win_path: str) -> str:
    """将 Windows 路径转换为 WSL 路径。"""
    if not win_path:
        return win_path

    normalized = win_path.strip().replace('"', "")
    if len(normalized) >= 2 and normalized[1] == ":":
        drive = normalized[0].lower()
        rest = normalized[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return normalized


def find_images(input_path: Path) -> Iterator[tuple[Path, Path]]:
    """递归查找 JPG 图片。"""
    for root, _, files in os.walk(input_path):
        for filename in files:
            src = Path(root) / filename
            if src.suffix.lower() in IMAGE_EXTENSIONS:
                yield src.relative_to(input_path), src


def resize_image(
    src: Path,
    dst: Path,
    target_size: tuple[int, int],
    quality: int,
    resize_mode: str,
    background_color: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """缩放单张 JPG 图片并尽量保持原始 JPEG 保存参数。"""
    with Image.open(src) as img:
        original_size = img.size
        original_quantization = getattr(img, "quantization", None)
        original_exif = img.getexif()
        icc_profile = img.info.get("icc_profile")
        dpi = img.info.get("dpi")
        progressive = bool(img.info.get("progressive") or img.info.get("progression"))

        try:
            subsampling = JpegImagePlugin.get_sampling(img)
        except Exception:
            subsampling = None

        transposed = ImageOps.exif_transpose(img)
        working = transposed.convert("RGB")
        resized = resize_canvas(working, target_size, resize_mode, background_color)

        save_kwargs: dict[str, object] = {
            "format": "JPEG",
            "optimize": False,
        }

        if original_quantization:
            save_kwargs["qtables"] = original_quantization
        else:
            save_kwargs["quality"] = quality

        if subsampling in (0, 1, 2):
            save_kwargs["subsampling"] = subsampling

        if progressive:
            save_kwargs["progressive"] = True

        if dpi:
            save_kwargs["dpi"] = dpi

        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        if original_exif:
            original_exif[ORIENTATION_TAG] = 1
            save_kwargs["exif"] = original_exif.tobytes()

        dst.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst, **save_kwargs)
        return original_size, resized.size


def resize_canvas(
    image: Image.Image,
    target_size: tuple[int, int],
    resize_mode: str,
    background_color: tuple[int, int, int],
) -> Image.Image:
    """根据模式将图片缩放到目标尺寸。"""
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"目标分辨率无效: {target_width}x{target_height}")

    src_width, src_height = image.size
    if src_width <= 0 or src_height <= 0:
        raise ValueError(f"原始分辨率无效: {src_width}x{src_height}")

    if resize_mode == "exact":
        return image.resize(target_size, Image.Resampling.LANCZOS)

    scale_x = target_width / src_width
    scale_y = target_height / src_height

    if resize_mode == "fit_pad":
        scale = min(scale_x, scale_y)
    elif resize_mode == "fit_crop":
        scale = max(scale_x, scale_y)
    else:
        raise ValueError(f"不支持的 resize_mode: {resize_mode}")

    resized_width = max(1, round(src_width * scale))
    resized_height = max(1, round(src_height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    if resize_mode == "fit_pad":
        canvas = Image.new("RGB", target_size, background_color)
        offset_x = (target_width - resized_width) // 2
        offset_y = (target_height - resized_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return resized.crop((left, top, left + target_width, top + target_height))


def _worker_wrapper(
    args: tuple[int, tuple[Path, Path]],
    target_size: tuple[int, int],
    quality: int,
    keep_structure: bool,
    output_path: Path,
    resize_mode: str,
    background_color: tuple[int, int, int],
) -> tuple[str, bool, str | None, tuple[int, int] | None, tuple[int, int] | None]:
    """多进程包装函数。"""
    idx, (rel, src) = args
    try:
        if keep_structure:
            dst = output_path / rel
        else:
            dst = output_path / f"{idx:06d}{src.suffix.lower()}"

        original_size, new_size = resize_image(
            src=src,
            dst=dst,
            target_size=target_size,
            quality=quality,
            resize_mode=resize_mode,
            background_color=background_color,
        )
        return (src.name, True, None, original_size, new_size)
    except Exception as exc:
        return (src.name, False, str(exc), None, None)


def process_files(
    input_path: Path,
    output_path: Path,
    target_size: tuple[int, int],
    quality: int,
    keep_structure: bool,
    preview: bool,
    preview_limit: int,
    workers: int | None,
    resize_mode: str,
    background_color: tuple[int, int, int],
) -> None:
    """批量处理主逻辑。"""
    if not input_path.exists():
        print(f"[-] 输入路径不存在: {input_path}")
        sys.exit(1)

    if input_path.resolve() == output_path.resolve():
        print("[-] 输入路径和输出路径不能相同，请输出到新目录")
        sys.exit(1)

    files = list(find_images(input_path))
    if not files:
        print("[-] 未找到 JPG 图片文件")
        return

    print("=" * 80)
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"目标分辨率: {target_size[0]}x{target_size[1]}")
    print(f"缩放模式: {resize_mode}")
    print(f"保持目录结构: {keep_structure}")
    print(f"JPEG 回退质量: {quality}")
    print(f"待处理文件数: {len(files)}")

    if preview:
        print("\n" + "=" * 80)
        print(f"预览模式 (总计: {len(files)} | 显示前 {preview_limit} 个)")
        print("=" * 80)
        for idx, (rel, src) in enumerate(files[:preview_limit], 1):
            dst = output_path / (
                rel if keep_structure else f"{idx:06d}{src.suffix.lower()}"
            )
            with Image.open(src) as img:
                preview_size = img.size

            print("\n[操作: 调整 JPG 分辨率]")
            print(f"  源: {src}")
            print(f"  原尺寸: {preview_size[0]}x{preview_size[1]}")
            print(f"  到: {dst}")
            print(f"  目标尺寸: {target_size[0]}x{target_size[1]}")
            print(f"  参数: mode={resize_mode}, fallback_quality={quality}")

        print("\n" + "=" * 80)
        if input("确认执行? (y/n): ").strip().lower() != "y":
            print("[!] 已取消")
            return

    output_path.mkdir(parents=True, exist_ok=True)
    worker_count = workers or cpu_count()
    print(f"[*] 启动 {worker_count} 个进程处理 {len(files)} 个文件...")

    args_list = list(enumerate(files, 1))
    worker_func = partial(
        _worker_wrapper,
        target_size=target_size,
        quality=quality,
        keep_structure=keep_structure,
        output_path=output_path,
        resize_mode=resize_mode,
        background_color=background_color,
    )

    ok = 0
    fail = 0
    failed_log: list[tuple[str, str]] = []

    with Pool(processes=worker_count) as pool:
        for index, (name, success, error, original_size, new_size) in enumerate(
            pool.imap_unordered(worker_func, args_list),
            1,
        ):
            if success and original_size and new_size:
                print(
                    f"[{index}/{len(files)}] ✓ {name} "
                    f"({original_size[0]}x{original_size[1]} -> {new_size[0]}x{new_size[1]})"
                )
                ok += 1
            else:
                print(f"[{index}/{len(files)}] ✗ {name}: {error}")
                fail += 1
                failed_log.append((name, error or "unknown error"))

    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    if failed_log:
        log_file = output_path / "_resize_failed.log"
        with open(log_file, "w", encoding="utf-8") as file:
            for name, error in failed_log:
                file.write(f"{name}: {error}\n")
        print(f"[!] 失败详情已记录至: {log_file}")


def main() -> None:
    # ============ 配置参数 (直接修改此处) ============
    INPUT = path_win2wsl(r"F:\dataset\select_clein_reblad_data")
    OUTPUT = path_win2wsl(r"F:\dataset\select_clein_reblad_data_resized")
    TARGET_SIZE = (3072 // 2, 1024 // 2)
    QUALITY = 99
    RESIZE_MODE = "exact"
    BACKGROUND_COLOR = (255, 255, 255)
    KEEP_STRUCTURE = True
    PREVIEW = True
    PREVIEW_LIMIT = 5
    WORKERS = None
    # ===============================================

    process_files(
        input_path=Path(INPUT),
        output_path=Path(OUTPUT),
        target_size=TARGET_SIZE,
        quality=QUALITY,
        keep_structure=KEEP_STRUCTURE,
        preview=PREVIEW,
        preview_limit=PREVIEW_LIMIT,
        workers=WORKERS,
        resize_mode=RESIZE_MODE,
        background_color=BACKGROUND_COLOR,
    )


if __name__ == "__main__":
    main()
