#!/usr/bin/env python3
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image

# 防止处理大图片时报错
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def path_win2wsl(win_path: str) -> str:
    """Windows路径转WSL路径"""
    if not win_path:
        return win_path
    win_path = win_path.strip()
    # 移除引号
    win_path = win_path.replace('"', '').replace("'", "")
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path


def find_images(input_path: Path):
    """递归查找图片文件"""
    for root, _, files in os.walk(input_path):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p.relative_to(input_path), p


def _process_single(src: Path, dst: Path, scale: float, quality: int = 100):
    """
    处理单个文件:
    1. 切割成4等份 (1, 2, 3, 4)
    2. 重组为 (3, 2, 4)
    3. 缩放
    """
    try:
        with Image.open(src) as img:
            w, h = img.size
            part_w = w // 4
            
            # 1. 切割: 0-indexed implies parts[0] is '1', parts[1] is '2', etc.
            # Parts indices: 0=1, 1=2, 2=3, 3=4
            parts = []
            for i in range(4):
                box = (i * part_w, 0, (i + 1) * part_w, h)
                parts.append(img.crop(box))
            
            # 2. 重组: 3, 2, 4 -> Indices: 2, 1, 3
            target_indices = [2, 1, 3] # Part 3, Part 2, Part 4
            new_parts = [parts[i] for i in target_indices]
            
            new_w_full = part_w * 3
            new_h_full = h
            
            # 拼接
            combined = Image.new(img.mode, (new_w_full, new_h_full))
            for i, part in enumerate(new_parts):
                combined.paste(part, (i * part_w, 0))
            
            # 3. 缩放
            if scale != 1.0:
                target_w = int(new_w_full * scale)
                target_h = int(new_h_full * scale)
                # 使用 LANCZOS 进行高质量缩放
                final_img = combined.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                final_img = combined

            # 保存
            dst.parent.mkdir(parents=True, exist_ok=True)
            # 如果是JPEG，使用quality参数；如果是PNG，quality参数可能被忽略但不会报错
            if dst.suffix.lower() in ['.jpg', '.jpeg']:
                final_img.save(dst, quality=quality, subsampling=0)
            else:
                final_img.save(dst, quality=quality)
                
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {e}")


def _process_worker(args, scale, quality, keep_structure):
    """多进程工作函数"""
    _idx, (rel, src), output_path = args
    try:
        if keep_structure:
            dst = output_path / rel
        else:
            # 如果不保持结构，可以在这里自定义命名规则，这里暂且保持原名以便对照
            # 或者改为 output_path / f"{src.stem}_reconstructed{src.suffix}"
            dst = output_path / rel
            
        _process_single(src, dst, scale, quality)
        return (src.name, True, None)
    except Exception as e:
        return (src.name, False, str(e))


def process_files(input_path: Path, output_path: Path, scale: float,
                  quality: int = 100,
                  keep_structure: bool = True,
                  preview: bool = False, preview_limit: int = 5,
                  workers: int = None):
    files = list(find_images(input_path))
    if not files:
        print("未找到图片")
        return

    # 预览
    if preview:
        print("=" * 80)
        print(f"预览模式 (共 {len(files)} 个文件, 显示前 {preview_limit} 个):")
        print("=" * 80)
        for _idx, (rel, src) in enumerate(files[:preview_limit], start=1):
            if keep_structure:
                dst = output_path / rel
            else:
                dst = output_path / rel
            print("\n[操作: 切割(4)->重组(3,2,4)->缩放]")
            print(f"  源: {src}")
            print(f"  到: {dst}")
            print(f"  参数: scale={scale}, quality={quality}")
        if len(files) > preview_limit:
            print(f"\n... 还有 {len(files) - preview_limit} 个文件 ...")
        print("=" * 80)
        # 非交互环境下如果要直接运行，可以将此处改为自动确认，或者在main中控制PREVIEW=False
        if input("\n确认执行? (y/n): ").lower() != "y":
            print("已取消")
            return

    # 多进程处理
    worker_count = workers if workers else cpu_count()
    print(f"使用 {worker_count} 个进程处理 {len(files)} 个文件...")
    
    output_path.mkdir(parents=True, exist_ok=True)
    args_list = [(idx, item, output_path) for idx, item in enumerate(files, start=1)]
    worker_func = partial(_process_worker, scale=scale, quality=quality, keep_structure=keep_structure)
    
    ok = fail = 0
    failed_files = []
    
    # 使用 imap_unordered 实时显示进度
    with Pool(processes=worker_count) as pool:
        results = pool.imap_unordered(worker_func, args_list)
        
        for i, (name, success, error) in enumerate(results, 1):
            if success:
                print(f"✓ {name} ({i}/{len(files)})")
                ok += 1
            else:
                print(f"✗ {name}: {error} ({i}/{len(files)})")
                fail += 1
                failed_files.append((name, error))

    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    
    if failed_files:
        log_path = output_path / "_process_failed.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# 处理失败记录\n")
            f.write(f"# 总计: {fail} 个文件失败\n\n")
            for name, error in failed_files:
                f.write(f"{name}: {error}\n")
        print(f"\n失败文件列表已保存到: {log_path}")


def main():
    # ============ 配置参数 ============
    INPUT = path_win2wsl(r"F:\dataset\select_clein")          # 输入文件夹 (请修改此处)
    OUTPUT = path_win2wsl(r"F:\dataset\select_clein_lq_r")    # 输出文件夹 (请修改此处)
    SCALE = 0.5                            # 缩放比例 (1024*4 x 1024 -> 1536x512, 即0.5)
    QUALITY = 100                          # 图片质量 (1-100)
    KEEP_STRUCTURE = True                  # 保持原文件夹结构
    PREVIEW = True                         # 预览模式 (确认无误后可改为False直接运行)
    PREVIEW_LIMIT = 5                      # 预览条数
    WORKERS = None                         # 进程数, None=自动
    # ==================================

    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        print("请在脚本 main 函数中修改 INPUT 路径。")
        return

    process_files(
        input_path=input_path,
        output_path=output_path,
        scale=SCALE,
        quality=QUALITY,
        keep_structure=KEEP_STRUCTURE,
        preview=PREVIEW,
        preview_limit=PREVIEW_LIMIT,
        workers=WORKERS,
    )


if __name__ == "__main__":
    main()
