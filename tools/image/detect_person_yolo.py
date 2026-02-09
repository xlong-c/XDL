#!/usr/bin/env python3
"""使用 YOLO 批量检测图片中的人物，并导出 CSV 结果。"""
import os
import sys
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFile, ImageFont

# 针对图片处理，添加截断处理
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from ultralytics import YOLO
except ImportError:
    print("[-] 请先安装 ultralytics: pip install ultralytics")
    sys.exit(1)

# 支持的图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

# 全局模型实例，避免在多进程中重复加载（如果在 fork 模式下有效）
# 注意：在 Windows 或 WSL 的 spawn 模式下，子进程仍会加载
_MODEL_CACHE = {}

def get_model(model_path: str):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]

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

def load_font(size: int = 20) -> ImageFont.ImageFont:
    """优先使用 Arial，不可用时降级为默认字体。"""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

def _process_single(src: Path, model_path: str, conf_threshold: float, end2end: bool) -> List[dict]:
    """
    单文件处理核心逻辑 (检测人物)
    """
    model = get_model(model_path)
    predictions = model.predict(
        str(src),
        conf=conf_threshold,
        verbose=False,
        end2end=end2end,
        classes=[0],
    )

    detections = []
    if predictions and predictions[0].boxes is not None:
        boxes = predictions[0].boxes
        box_xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        for idx, (box, conf) in enumerate(zip(box_xyxy, confidences)):
            detections.append({
                "box_id": idx,
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "confidence": float(conf)
            })
    return detections

def _worker_wrapper(args, model_path, conf_threshold, end2end, output_path, debug_idx):
    """多进程包装函数"""
    idx, (rel, src) = args
    try:
        detections = _process_single(src, model_path, conf_threshold, end2end)
        
        # 如果是调试索引，则绘制并保存图片
        debug_output = None
        if idx == debug_idx:
            output_path.mkdir(parents=True, exist_ok=True)
            debug_output = output_path / f"debug_{idx:04d}_{src.name}"
            
            with Image.open(src) as img:
                draw = ImageDraw.Draw(img)
                font = load_font()
                for det in detections:
                    draw.rectangle([det["x1"], det["y1"], det["x2"], det["y2"]], outline="red", width=3)
                    label_y = max(0.0, det["y1"] - 20)
                    draw.text((det["x1"], label_y), f"{det['confidence']:.2f}", fill="red", font=font)
                img.save(debug_output, quality=99)

        return (src.name, rel, True, detections, None, debug_output)
    except Exception as e:
        return (src.name, rel, False, [], str(e), None)

def process_files(input_path: Path, output_path: Path, **config):
    """批量处理主逻辑"""
    files = list(find_images(input_path))
    if not files:
        print("[-] 未找到图片文件")
        return

    # 1. 预览模式
    if config.get('PREVIEW'):
        print("=" * 80)
        print(f"预览模式 (总计: {len(files)} | 显示前 {config.get('PREVIEW_LIMIT')} 个)")
        print("=" * 80)
        for idx, (rel, src) in enumerate(files[:config.get('PREVIEW_LIMIT')], 1):
            print(f"\n[操作: 检测人物]")
            print(f"  源: {src}")
            print(f"  模型: {config.get('MODEL_PATH')}")
            print(f"  阈值: {config.get('CONF_THRESHOLD')}, End2End: {config.get('END2END')}")
        
        print("\n" + "=" * 80)
        if input("确认执行? (y/n): ").lower() != "y":
            print("[!] 已取消")
            return

    # 2. 执行处理 (多进程)
    output_path.mkdir(parents=True, exist_ok=True)
    worker_count = config.get('WORKERS') or cpu_count()
    print(f"[*] 启动 {worker_count} 个进程处理 {len(files)} 个文件...")
    
    args_list = list(enumerate(files, 1))
    worker_func = partial(
        _worker_wrapper, 
        model_path=config.get('MODEL_PATH'), 
        conf_threshold=config.get('CONF_THRESHOLD'),
        end2end=config.get('END2END'),
        output_path=output_path,
        debug_idx=config.get('DEBUG_IMAGE_IDX')
    )
    
    ok = fail = 0
    all_results = []
    failed_log = []
    
    with Pool(processes=worker_count) as pool:
        for i, (name, rel, success, detections, error, debug_out) in enumerate(pool.imap_unordered(worker_func, args_list), 1):
            if success:
                print(f"[{i}/{len(files)}] ✓ {name} (检测到 {len(detections)} 个人)")
                ok += 1
                all_results.append({
                    "name": name,
                    "rel": rel,
                    "detections": detections,
                    "src": input_path / rel
                })
                if debug_out:
                    print(f"    [!] 调试图片已保存: {debug_out}")
            else:
                print(f"[{i}/{len(files)}] ✗ {name}: {error}")
                fail += 1
                failed_log.append((name, error))

    # 3. 统计与日志
    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    
    # 导出 CSV
    if all_results:
        csv_path = output_path / "detection_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_path", "relative_path", "person_count", "box_id", "x1", "y1", "x2", "y2", "confidence"])
            for res in all_results:
                if res["detections"]:
                    for det in res["detections"]:
                        writer.writerow([
                            str(res["src"]), str(res["rel"]), len(res["detections"]),
                            det["box_id"], det["x1"], det["y1"], det["x2"], det["y2"], det["confidence"]
                        ])
                else:
                    writer.writerow([str(res["src"]), str(res["rel"]), 0, "", "", "", "", "", ""])
        print(f"[*] CSV结果已保存至: {csv_path}")

    if failed_log:
        log_file = output_path / "_process_failed.log"
        with open(log_file, "w", encoding="utf-8") as f:
            for name, err in failed_log: f.write(f"{name}: {err}\n")
        print(f"[!] 失败详情已记录至: {log_file}")

def main():
    # ============ 配置参数 (直接修改此处) ============
    INPUT  = path_win2wsl(r"F:\raw_pics\cloths\773")    # 输入路径
    OUTPUT = path_win2wsl(r"F:\raw_pics\cloths")        # 输出路径
    MODEL_PATH = "downloads/yolo26m.pt"                  # 模型路径
    CONF_THRESHOLD = 0.75                               # 置信度阈值
    END2END = True                                      # 是否使用 End2End 模式
    PREVIEW = True                                      # 开启预览确认
    PREVIEW_LIMIT = 5                                   # 预览显示条数
    WORKERS = None                                      # 进程数 (None 为自动)
    DEBUG_IMAGE_IDX = 0                                 # 调试图片序号 (1-N, 0为关闭)
    # ===============================================

    process_files(
        input_path=Path(INPUT),
        output_path=Path(OUTPUT),
        MODEL_PATH=MODEL_PATH,
        CONF_THRESHOLD=CONF_THRESHOLD,
        END2END=END2END,
        PREVIEW=PREVIEW,
        PREVIEW_LIMIT=PREVIEW_LIMIT,
        WORKERS=WORKERS,
        DEBUG_IMAGE_IDX=DEBUG_IMAGE_IDX
    )

if __name__ == "__main__":
    main()
