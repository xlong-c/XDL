#!/usr/bin/env python3
"""
批量处理图片文件:递归查找、复制/迁移、可选转换为JPG
直接修改下方配置参数后运行
"""

import os
import sys
import shutil
from pathlib import Path
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".webp",
    ".raw",
    ".cr2",
    ".nef",
}


def path_win2wsl(win_path: str) -> str:
    if not win_path:
        return win_path
    win_path = win_path.strip()
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path


def is_image_file(filepath: Path) -> bool:
    ext = filepath.suffix.lower()
    return ext in IMAGE_EXTENSIONS


def find_all_images(input_path: Path) -> list:
    images = []
    input_path = input_path.resolve()

    for root, dirs, files in os.walk(input_path):
        for filename in files:
            filepath = Path(root) / filename
            if is_image_file(filepath):
                rel_path = filepath.relative_to(input_path)
                images.append((rel_path, filepath))

    return images


def convert_to_jpg(src_path: Path, dst_path: Path, quality: int = 99) -> bool:
    try:
        with Image.open(src_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if img.mode in ("RGBA", "LA"):
                    background.paste(
                        img,
                        mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None,
                    )
                    img = background
                else:
                    img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            if not str(dst_path).lower().endswith(".jpg"):
                dst_path = dst_path.with_suffix(".jpg")

            img.save(dst_path, "JPEG", quality=quality, optimize=True)
            return True
    except Exception as e:
        print(f"转换失败 {src_path}: {e}")
        return False


def process_images(
    input_path: Path,
    output_path: Path,
    keep_structure: bool = False,
    convert_jpg: bool = False,
    jpg_quality: int = 99,
    move: bool = False,
    prefix: str = "",
    start: int = 1,
    digits: int = 5,
    rename: bool = True,
    preview: bool = True,
    preview_limit: int = 5,
):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)

    print("=" * 80)
    print("配置信息:")
    print(f"  输入路径:  {input_path}")
    print(f"  输出路径:  {output_path}")
    print(f"  操作模式:  {'迁移' if move else '复制'}")
    print(f"  文件夹布局: {'保持原结构' if keep_structure else '平铺'}")
    print(
        f"  格式转换:  {'转换为JPG (质量:' + str(jpg_quality) + ')' if convert_jpg else '保持原格式'}"
    )
    if not keep_structure:
        if rename:
            print(
                f"  命名规则:  {prefix + '_' if prefix else ''}{start:0{digits}d} ... {prefix + '_' if prefix else ''}{(start + 999):0{digits}d}"
            )
        else:
            print(f"  命名规则:  保持原文件名")
    print("=" * 80)

    print(f"\n正在扫描: {input_path}")
    images = find_all_images(input_path)

    if not images:
        print("未找到图片文件")
        return

    print(f"找到 {len(images)} 个图片文件\n")

    # 预览模式
    if preview:
        print("-" * 80)
        print(
            f"预览模式 (显示前 {min(preview_limit, len(images))} 条, 共 {len(images)} 条):"
        )
        print("-" * 80)

        for idx, (rel_path, src_path) in enumerate(images[:preview_limit], start=start):
            if keep_structure:
                dst_path = output_path / rel_path
                if convert_jpg:
                    dst_path = dst_path.with_suffix(".jpg")
            else:
                if rename:
                    ext = ".jpg" if convert_jpg else src_path.suffix
                    if prefix:
                        new_name = f"{prefix}_{idx:0{digits}d}{ext}"
                    else:
                        new_name = f"{idx:0{digits}d}{ext}"
                    dst_path = output_path / new_name
                else:
                    dst_path = output_path / src_path.name

            action = "迁移" if move else "复制"
            if convert_jpg and src_path.suffix.lower() not in (".jpg", ".jpeg"):
                action += "+转换JPG"

            print(f"{action}: {src_path}")
            print(f"      -> {dst_path}")
            print()

        if len(images) > preview_limit:
            print(f"... 还有 {len(images) - preview_limit} 个文件未显示 ...\n")

        confirm = input("确认执行? (y/n): ")
        if confirm.lower() != "y":
            print("已取消")
            return
        print()

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_count = 0

    for idx, (rel_path, src_path) in enumerate(images, start=start):
        try:
            if keep_structure:
                dst_path = output_path / rel_path
                if convert_jpg:
                    dst_path = dst_path.with_suffix(".jpg")
                dst_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                if rename:
                    ext = ".jpg" if convert_jpg else src_path.suffix
                    if prefix:
                        new_name = f"{prefix}_{idx:0{digits}d}{ext}"
                    else:
                        new_name = f"{idx:0{digits}d}{ext}"
                    dst_path = output_path / new_name
                else:
                    dst_path = output_path / src_path.name

            if dst_path.exists():
                print(f"跳过(已存在): {dst_path}")
                continue

            if convert_jpg and src_path.suffix.lower() not in (".jpg", ".jpeg"):
                if convert_to_jpg(src_path, dst_path, jpg_quality):
                    print(f"转换: {src_path.name} -> {dst_path.name}")
                    success_count += 1
                else:
                    failed_count += 1
                    continue
            else:
                shutil.copy2(src_path, dst_path)
                print(f"复制: {src_path.name} -> {dst_path.name}")
                success_count += 1

            if move:
                src_path.unlink()

        except Exception as e:
            print(f"处理失败 {rel_path}: {e}")
            failed_count += 1

    print("\n" + "=" * 80)
    print(f"完成! 成功: {success_count}, 失败: {failed_count}")
    if move:
        print("已删除源文件(迁移模式)")
    print("=" * 80)


def main():
    # ==================== 配置参数 ====================
    # 路径设置 (Windows路径会自动转换为WSL路径)
    INPUT_PATH = r"F:\dataset\select_zbr\img_z"  # 输入路径:包含图片的源文件夹
    OUTPUT_PATH = r"F:\dataset\select_zbr\img_z_all"  # 输出路径:目标文件夹

    # 转换为WSL路径
    INPUT_PATH = path_win2wsl(INPUT_PATH)
    OUTPUT_PATH = path_win2wsl(OUTPUT_PATH)

    # 操作模式
    KEEP_STRUCTURE = False  # 文件夹布局:True=保持原结构,False=平铺
    MOVE_MODE = False  # 操作模式:True=迁移(删除源文件),False=复制
    CONVERT_JPG = False  # 格式转换:True=转JPG,False=保持原格式
    JPG_QUALITY = 99  # JPG质量:1-100

    # 命名规则(仅平铺模式有效)
    RENAME = False
    PREFIX = ""  # 文件名前缀
    START_NUM = 1  # 起始序号
    DIGITS = 6  # 序号位数

    # 预览设置
    PREVIEW = True  # 预览模式:True=先预览再确认,False=直接执行
    PREVIEW_LIMIT = 5  # 预览显示条数
    # =================================================

    process_images(
        input_path=Path(INPUT_PATH),
        output_path=Path(OUTPUT_PATH),
        keep_structure=KEEP_STRUCTURE,
        convert_jpg=CONVERT_JPG,
        jpg_quality=JPG_QUALITY,
        move=MOVE_MODE,
        prefix=PREFIX,
        start=START_NUM,
        digits=DIGITS,
        rename=RENAME,
        preview=PREVIEW,
        preview_limit=PREVIEW_LIMIT,
    )


if __name__ == "__main__":
    main()
