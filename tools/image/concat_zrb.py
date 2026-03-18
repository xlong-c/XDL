#!/usr/bin/env python3
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image, ImageFile

# 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 支持的图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def path_win2wsl(win_path: str) -> str:
    """将 Windows 路径转换为 WSL 路径"""
    if not win_path: return win_path
    win_path = win_path.strip().replace('"', '')
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path

def find_images(input_path: Path):
    """递归查找所有图片文件"""
    for root, _, files in os.walk(input_path):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p.relative_to(input_path), p

def _process_single(src_b: Path, src_z: Path, src_r: Path, dst: Path, quality: int):
    """
    单文件处理核心逻辑 (子进程执行)
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # 打开图像并统一转换为 RGB 模式
    img_b = Image.open(src_b).convert("RGB")
    img_z = Image.open(src_z).convert("RGB")
    img_r = Image.open(src_r).convert("RGB")
    
    # Resize 到 512x512
    img_b = img_b.resize((512, 512), Image.Resampling.LANCZOS)
    img_z = img_z.resize((512, 512), Image.Resampling.LANCZOS)
    img_r = img_r.resize((512, 512), Image.Resampling.LANCZOS)
    
    # 按照 z -> r -> b 的顺序横向拼接 (总分辨率 1536x512)
    result = Image.new("RGB", (1536, 512))
    result.paste(img_z, (0, 0))
    result.paste(img_r, (512, 0))
    result.paste(img_b, (1024, 0))
    
    # 保存结果
    result.save(dst, quality=quality)

def _worker_wrapper(args, quality, output_path, dir_z, dir_r):
    """多进程包装函数"""
    idx, (rel, src_b) = args
    try:
        name = src_b.name
        # 从文件名提取 idx，例如：12_00001.jpg -> 12
        base_idx = name.split('_')[0]
        
        # 匹配 z 图像 (可能是 12_00001.jpg 或 12.jpg)
        z_cand1 = dir_z / name
        z_cand2 = dir_z / f"{base_idx}{src_b.suffix}"
        z_cand3 = dir_z / f"{base_idx}.jpg"
        src_z = z_cand1 if z_cand1.exists() else (z_cand2 if z_cand2.exists() else z_cand3)
        if not src_z.exists():
            return (name, False, f"Missing Z image: {z_cand1} or {z_cand3}")
            
        # 匹配 r 图像 (可能是 12.jpg 或 12_00001.jpg)
        r_cand1 = dir_r / f"{base_idx}.jpg"
        r_cand2 = dir_r / f"{base_idx}{src_b.suffix}"
        r_cand3 = dir_r / name
        src_r = r_cand1 if r_cand1.exists() else (r_cand2 if r_cand2.exists() else r_cand3)
        if not src_r.exists():
            return (name, False, f"Missing R image: {r_cand1} or {r_cand3}")

        dst = output_path / name
        
        _process_single(src_b, src_z, src_r, dst, quality)
        return (name, True, None)
    except Exception as e:
        return (args[1][1].name, False, str(e))

def process_files(base_path: Path, output_path: Path, **config):
    """批量处理主逻辑"""
    dir_b = base_path / "img_b"
    dir_z = base_path / "img_z"
    dir_r = base_path / "img_r"
    
    if not dir_b.exists():
        print(f"[-] 找不到文件夹: {dir_b}")
        return

    files_b = list(find_images(dir_b))
    if not files_b:
        print("[-] 在 img_b 中未找到图片文件")
        return

    # 1. 预览模式
    if config.get('PREVIEW'):
        print("=" * 80)
        print(f"预览模式 (img_b 总计: {len(files_b)} | 显示前 {config.get('PREVIEW_LIMIT')} 个)")
        print("=" * 80)
        for idx, (rel, src_b) in enumerate(files_b[:config.get('PREVIEW_LIMIT')], 1):
            name = src_b.name
            base_idx = name.split('_')[0]
            dst = output_path / name
            
            z_cand1 = dir_z / name
            z_cand2 = dir_z / f"{base_idx}{src_b.suffix}"
            z_cand3 = dir_z / f"{base_idx}.jpg"
            src_z = z_cand1 if z_cand1.exists() else (z_cand2 if z_cand2.exists() else z_cand3)
            
            r_cand1 = dir_r / f"{base_idx}.jpg"
            r_cand2 = dir_r / f"{base_idx}{src_b.suffix}"
            r_cand3 = dir_r / name
            src_r = r_cand1 if r_cand1.exists() else (r_cand2 if r_cand2.exists() else r_cand3)
            
            print(f"\n[操作: 拼接 z -> r -> b]")
            print(f"  img_z: {src_z} ({'存在' if src_z.exists() else '缺失!'})")
            print(f"  img_r: {src_r} ({'存在' if src_r.exists() else '缺失!'})")
            print(f"  img_b: {src_b} (存在)")
            print(f"  到:    {dst}")
            print(f"  Quality: {config.get('QUALITY')}")
        
        print("\n" + "=" * 80)
        if input("确认执行? (y/n): ").lower() != "y":
            print("[!] 已取消")
            return

    # 2. 执行处理 (多进程)
    output_path.mkdir(parents=True, exist_ok=True)
    worker_count = config.get('WORKERS') or cpu_count()
    print(f"[*] 启动 {worker_count} 个进程处理 {len(files_b)} 个文件...")
    
    args_list = list(enumerate(files_b, 1))
    worker_func = partial(
        _worker_wrapper, 
        quality=config.get('QUALITY'),
        output_path=output_path,
        dir_z=dir_z,
        dir_r=dir_r
    )
    
    ok = fail = 0
    failed_log = []
    
    with Pool(processes=worker_count) as pool:
        for i, (name, success, error) in enumerate(pool.imap_unordered(worker_func, args_list), 1):
            if success:
                print(f"[{i}/{len(files_b)}] ✓ {name}")
                ok += 1
            else:
                print(f"[{i}/{len(files_b)}] ✗ {name}: {error}")
                fail += 1
                failed_log.append((name, error))

    # 3. 统计与日志
    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    if failed_log:
        log_file = output_path / "_process_failed.log"
        with open(log_file, "w", encoding="utf-8") as f:
            for name, err in failed_log: f.write(f"{name}: {err}\n")
        print(f"[!] 失败详情已记录至: {log_file}")

def main():
    # ============ 配置参数 (直接修改此处) ============
    BASE_DIR = path_win2wsl(r"F:\dataset\select_zbr")        # 基础路径，包含 img_z, img_b, img_r
    OUTPUT   = path_win2wsl(r"F:\dataset\select_zbr\concat") # 输出路径
    QUALITY  = 99                             # 保存质量 (1-100)
    PREVIEW  = True                           # 开启预览确认
    PREVIEW_LIMIT = 5                         # 预览显示条数
    WORKERS  = None                           # 进程数 (None 为自动)
    # ===============================================

    process_files(
        base_path=Path(BASE_DIR),
        output_path=Path(OUTPUT),
        QUALITY=QUALITY,
        PREVIEW=PREVIEW,
        PREVIEW_LIMIT=PREVIEW_LIMIT,
        WORKERS=WORKERS
    )

if __name__ == "__main__":
    main()
