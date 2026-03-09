#!/usr/bin/env python3
from __future__ import annotations

import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def path_win2wsl(win_path: str) -> str:
    """将 Windows 路径转换为 WSL 路径。"""
    if not win_path:
        return win_path
    clean_path = win_path.strip().replace('"', "").replace("'", "")
    if len(clean_path) >= 2 and clean_path[1] == ":":
        drive = clean_path[0].lower()
        rest = clean_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return clean_path


def path_wsl2win(wsl_path: str) -> str:
    """将 WSL 路径转换为 Windows 路径。"""
    if not wsl_path:
        return wsl_path
    clean_path = wsl_path.strip().replace('"', "").replace("'", "")
    if clean_path.startswith("/mnt/") and len(clean_path) > 6:
        drive = clean_path[5].upper()
        rest = clean_path[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    return clean_path


def normalize_path(path_str: str) -> Path:
    """根据当前系统规范化路径。"""
    if os.name == "nt":
        return Path(path_wsl2win(path_str))
    return Path(path_win2wsl(path_str))


def find_images(input_path: Path) -> Iterable[Tuple[Path, Path]]:
    """递归查找图片文件。"""
    for root, _, files in os.walk(input_path):
        for file_name in files:
            src = Path(root) / file_name
            if src.suffix.lower() in IMAGE_EXTENSIONS:
                yield src.relative_to(input_path), src


def build_b_index(input_path_b: Path) -> Dict[str, List[Path]]:
    """为文件夹 B 建立基于主文件名的索引。"""
    index: Dict[str, List[Path]] = {}
    for _, src in find_images(input_path_b):
        match_key = src.stem.split("_")[0]
        index.setdefault(match_key, []).append(src)
    for paths in index.values():
        paths.sort()
    return index


def choose_b_file(
    a_src: Path,
    candidates: Sequence[Path],
    preferred_suffix: str,
) -> Optional[Path]:
    """根据 A 文件名挑选唯一匹配的 B 文件。"""
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    if preferred_suffix:
        expected_stem = f"{a_src.stem}{preferred_suffix}"
        for candidate in candidates:
            if candidate.stem == expected_stem:
                return candidate

    exact_matches = [candidate for candidate in candidates if candidate.stem == a_src.stem]
    if len(exact_matches) == 1:
        return exact_matches[0]

    raise RuntimeError(
        f"匹配到多个 B 文件，无法唯一确定: {a_src.name} -> {[path.name for path in candidates]}"
    )


def build_tasks(
    input_path_a: Path,
    input_path_b: Path,
    preferred_suffix: str,
) -> List[Tuple[int, Path, Path, Optional[Path]]]:
    """生成处理任务列表。"""
    files_a = list(find_images(input_path_a))
    b_index = build_b_index(input_path_b)
    tasks: List[Tuple[int, Path, Path, Optional[Path]]] = []

    for idx, (rel, a_src) in enumerate(files_a, 1):
        candidates = b_index.get(a_src.stem, [])
        try:
            b_src = choose_b_file(a_src, candidates, preferred_suffix)
        except RuntimeError:
            b_src = None
        tasks.append((idx, rel, a_src, b_src))

    return tasks


def parse_a_tile_key(tile_key: str) -> int:
    """将 A2/A3 解析为切片编号。"""
    if not tile_key.startswith("A"):
        raise ValueError(f"无效的 A 切片标识: {tile_key}")
    tile_index = int(tile_key[1:])
    if tile_index < 1 or tile_index > 4:
        raise ValueError(f"A 切片编号必须在 1 到 4 之间: {tile_key}")
    return tile_index


def build_output_path(output_path: Path, rel: Path, a_src: Path, keep_structure: bool) -> Path:
    """生成输出路径。"""
    if keep_structure:
        return (output_path / rel).with_suffix(".jpg")
    return output_path / f"{a_src.stem}.jpg"


def load_a_tiles(a_src: Path, tile_keys: Sequence[str]) -> Dict[str, Image.Image]:
    """从 A 图像中读取指定切片。"""
    required_indices = sorted({parse_a_tile_key(tile_key) for tile_key in tile_keys if tile_key.startswith("A")})
    if not required_indices:
        return {}

    with Image.open(a_src) as img_a:
        img_a = img_a.convert("RGB")
        width, height = img_a.size
        if width % 4 != 0:
            raise ValueError(f"A 图像宽度不能均分为 4 份: {a_src} -> {width}x{height}")
        tile_width = width // 4
        tiles: Dict[str, Image.Image] = {}
        for tile_index in required_indices:
            left = (tile_index - 1) * tile_width
            right = tile_index * tile_width
            tile = img_a.crop((left, 0, right, height)).copy()
            tiles[f"A{tile_index}"] = tile
        return tiles


def load_b_tile(b_src: Path, expected_size: Tuple[int, int], auto_resize_b: bool) -> Image.Image:
    """读取 B 图像并在需要时调整尺寸。"""
    with Image.open(b_src) as img_b:
        img_b = img_b.convert("RGB")
        if img_b.size != expected_size:
            if not auto_resize_b:
                raise ValueError(
                    f"B 图像尺寸不匹配: {b_src} -> {img_b.size}, 期望 {expected_size}"
                )
            img_b = img_b.resize(expected_size, Image.Resampling.LANCZOS)
        return img_b.copy()


def compose_output_image(
    a_src: Path,
    b_src: Optional[Path],
    output_layout: Sequence[str],
    auto_resize_b: bool,
) -> Image.Image:
    """将 A 的切片和 B 图像拼接为 3072x1024 的输出图。"""
    if b_src is None:
        raise FileNotFoundError(f"未找到对应的 B 文件: {a_src.name}")

    a_tiles = load_a_tiles(a_src, output_layout)
    first_a_tile = next((tile for key, tile in a_tiles.items() if key.startswith("A")), None)
    if first_a_tile is None:
        raise ValueError("输出布局中至少需要一个 A 切片")

    tile_size = first_a_tile.size
    b_tile = load_b_tile(b_src, tile_size, auto_resize_b)

    tile_map: Dict[str, Image.Image] = {**a_tiles, "B": b_tile}
    ordered_tiles: List[Image.Image] = []
    for tile_key in output_layout:
        if tile_key not in tile_map:
            raise ValueError(f"输出布局包含未知项: {tile_key}")
        ordered_tiles.append(tile_map[tile_key])

    widths = [tile.width for tile in ordered_tiles]
    heights = [tile.height for tile in ordered_tiles]
    if len(set(heights)) != 1:
        raise ValueError(f"待拼接图像高度不一致: {heights}")

    canvas = Image.new("RGB", (sum(widths), heights[0]))
    offset_x = 0
    for tile in ordered_tiles:
        canvas.paste(tile, (offset_x, 0))
        offset_x += tile.width
    return canvas


def _worker_wrapper(
    args: Tuple[int, Path, Path, Optional[Path]],
    output_path: Path,
    quality: int,
    keep_structure: bool,
    output_layout: Sequence[str],
    auto_resize_b: bool,
) -> Tuple[str, bool, Optional[str]]:
    """多进程包装函数。"""
    _, rel, a_src, b_src = args
    try:
        dst = build_output_path(output_path, rel, a_src, keep_structure)
        dst.parent.mkdir(parents=True, exist_ok=True)
        result = compose_output_image(a_src, b_src, output_layout, auto_resize_b)
        result.save(dst, format="JPEG", quality=quality, subsampling=0)
        return a_src.name, True, None
    except Exception as exc:
        return a_src.name, False, str(exc)


def process_files(
    input_path_a: Path,
    input_path_b: Path,
    output_path: Path,
    preferred_suffix: str,
    quality: int,
    keep_structure: bool,
    output_layout: Sequence[str],
    auto_resize_b: bool,
    preview: bool,
    preview_limit: int,
    workers: Optional[int],
) -> None:
    """批量处理主逻辑。"""
    if not input_path_a.exists():
        print(f"[-] 文件夹 A 不存在: {input_path_a}")
        return
    if not input_path_b.exists():
        print(f"[-] 文件夹 B 不存在: {input_path_b}")
        return

    tasks = build_tasks(input_path_a, input_path_b, preferred_suffix)
    if not tasks:
        print("[-] 未找到可处理的图片文件")
        return

    layout_text = " + ".join(output_layout)

    if preview:
        print("=" * 80)
        print(f"预览模式 (总计: {len(tasks)} | 显示前 {preview_limit} 个)")
        print("=" * 80)
        for idx, (task_index, rel, a_src, b_src) in enumerate(tasks[:preview_limit], 1):
            dst = build_output_path(output_path, rel, a_src, keep_structure)
            b_text = str(b_src) if b_src is not None else "<未匹配到对应文件>"
            print("\n[操作: 拼接 A 的中间切片与 B 图像]")
            print(f"  源 A: {a_src}")
            print(f"  源 B: {b_text}")
            print(f"  到: {dst}")
            print(f"  布局: {layout_text}")
            print(f"  参数: quality={quality}, task={task_index}")

        print("\n" + "=" * 80)
        if input("确认执行? (y/n): ").lower() != "y":
            print("[!] 已取消")
            return

    output_path.mkdir(parents=True, exist_ok=True)
    worker_count = workers or cpu_count()
    print(f"[*] 启动 {worker_count} 个进程处理 {len(tasks)} 个文件...")

    worker_func = partial(
        _worker_wrapper,
        output_path=output_path,
        quality=quality,
        keep_structure=keep_structure,
        output_layout=tuple(output_layout),
        auto_resize_b=auto_resize_b,
    )

    ok = 0
    fail = 0
    failed_log: List[Tuple[str, str]] = []

    with Pool(processes=worker_count) as pool:
        for i, (name, success, error) in enumerate(pool.imap_unordered(worker_func, tasks), 1):
            if success:
                print(f"[{i}/{len(tasks)}] ✓ {name}")
                ok += 1
            else:
                print(f"[{i}/{len(tasks)}] ✗ {name}: {error}")
                fail += 1
                failed_log.append((name, error or "未知错误"))

    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    if failed_log:
        log_file = output_path / "_process_failed.log"
        with open(log_file, "w", encoding="utf-8") as file_obj:
            for name, error in failed_log:
                file_obj.write(f"{name}: {error}\n")
        print(f"[!] 失败详情已记录至: {log_file}")


def main() -> None:
    INPUT_A = r"F:\dataset\select_clein"
    INPUT_B = r"F:\dataset\select_clein_reblad"
    OUTPUT = r"F:\dataset\select_clein_reblad_data"
    PREFERRED_B_STEM_SUFFIX = "_00001_"
    QUALITY = 99
    KEEP_STRUCTURE = False
    OUTPUT_LAYOUT = ("A2", "A3", "B")
    AUTO_RESIZE_B = False
    PREVIEW = True
    PREVIEW_LIMIT = 5
    WORKERS = None

    process_files(
        input_path_a=normalize_path(INPUT_A),
        input_path_b=normalize_path(INPUT_B),
        output_path=normalize_path(OUTPUT),
        preferred_suffix=PREFERRED_B_STEM_SUFFIX,
        quality=QUALITY,
        keep_structure=KEEP_STRUCTURE,
        output_layout=OUTPUT_LAYOUT,
        auto_resize_b=AUTO_RESIZE_B,
        preview=PREVIEW,
        preview_limit=PREVIEW_LIMIT,
        workers=WORKERS,
    )


if __name__ == "__main__":
    main()
