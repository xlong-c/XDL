#!/usr/bin/env python3
"""使用 YOLO 批量检测图片中的人物，并导出 CSV 结果。"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFile, ImageFont

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from ultralytics import YOLO
except ImportError:
    print("[-] 请先安装 ultralytics: pip install ultralytics")
    sys.exit(1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Detection:
    box_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass
class ImageResult:
    file_path: Path
    relative_path: Path
    detections: List[Detection]
    success: bool
    error: Optional[str] = None

    @property
    def person_count(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class Settings:
    input_path: str = r"F:\raw_pics\cloths\773"
    output_path: str = r"F:\raw_pics\cloths"
    model_path: str = "downloads/yolo26m.pt"
    conf_threshold: float = 0.75
    end2end: bool = True
    batch_size: int = 4
    preview: bool = True
    debug_image_idx: int = 0


def path_win2wsl(raw_path: str) -> str:
    """将 Windows 路径转换为 WSL 路径；非 Windows 路径保持不变。"""
    if not raw_path:
        return raw_path

    normalized = raw_path.strip().replace('"', "")
    if len(normalized) >= 2 and normalized[1] == ":":
        drive = normalized[0].lower()
        rest = normalized[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return normalized


def find_images(input_path: Path) -> Iterator[Tuple[Path, Path]]:
    """递归查找图片，返回 (相对路径, 绝对路径)。"""
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path.relative_to(input_path), path


def load_font(size: int = 20) -> ImageFont.ImageFont:
    """优先使用 Arial，不可用时降级为默认字体。"""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_boxes(
    image_path: Path, detections: Sequence[Detection], output_path: Path
) -> Path:
    """在图片上绘制检测框并保存。"""
    with Image.open(image_path) as image:
        draw = ImageDraw.Draw(image)
        font = load_font()

        for det in detections:
            draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline="red", width=3)
            label_y = max(0.0, det.y1 - 20)
            draw.text((det.x1, label_y), f"{det.confidence:.2f}", fill="red", font=font)

        image.save(output_path)
    return output_path


def detect_single(
    image_path: Path,
    model: YOLO,
    conf_threshold: float,
    end2end: bool,
) -> Tuple[List[Detection], Optional[str]]:
    """单张图片检测，仅保留 person 类别（class id = 0）。"""
    try:
        predictions = model.predict(
            str(image_path),
            conf=conf_threshold,
            verbose=False,
            end2end=end2end,
            classes=[0],
        )

        if not predictions:
            return [], None

        boxes = predictions[0].boxes
        if boxes is None:
            return [], None

        box_xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        detections = [
            Detection(
                box_id=idx,
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(conf),
            )
            for idx, (box, conf) in enumerate(zip(box_xyxy, confidences))
        ]
        return detections, None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def process_batch(
    image_paths: Sequence[Tuple[Path, Path]],
    model: YOLO,
    conf_threshold: float,
    end2end: bool,
) -> List[ImageResult]:
    """处理一批图片并输出结构化结果。"""
    batch_results: List[ImageResult] = []

    for rel_path, src_path in image_paths:
        detections, error = detect_single(src_path, model, conf_threshold, end2end)
        success = error is None

        if success:
            print(f"  ✓ {rel_path} - 检测到 {len(detections)} 个人物")
        else:
            print(f"  ✗ {rel_path}: {error}")

        batch_results.append(
            ImageResult(
                file_path=src_path,
                relative_path=rel_path,
                detections=detections,
                success=success,
                error=error,
            )
        )

    return batch_results


def export_csv(results: Sequence[ImageResult], output_path: Path) -> Path:
    """导出检测结果到 CSV。"""
    csv_path = output_path / "detection_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "file_path",
                "relative_path",
                "person_count",
                "box_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
            ]
        )

        for result in results:
            if result.detections:
                for det in result.detections:
                    writer.writerow(
                        [
                            str(result.file_path),
                            str(result.relative_path),
                            result.person_count,
                            det.box_id,
                            det.x1,
                            det.y1,
                            det.x2,
                            det.y2,
                            det.confidence,
                        ]
                    )
            else:
                writer.writerow(
                    [
                        str(result.file_path),
                        str(result.relative_path),
                        0,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )

    return csv_path


def debug_image(
    image_idx: int,
    files: Sequence[Tuple[Path, Path]],
    model: YOLO,
    conf_threshold: float,
    end2end: bool,
    output_path: Path,
) -> None:
    """调试指定序号图片，保存带框图片并打印详细检测结果。"""
    if not 1 <= image_idx <= len(files):
        print(f"[-] 无效的图片序号: {image_idx} (范围: 1-{len(files)})")
        return

    rel_path, src_path = files[image_idx - 1]
    print(f"\n[*] 调试图片 [{image_idx}]: {rel_path}")
    print(f"    完整路径: {src_path}")

    detections, error = detect_single(src_path, model, conf_threshold, end2end)
    if error:
        print(f"[-] 检测失败: {error}")
        return

    print(f"[*] 检测到 {len(detections)} 个人物:")
    for det in detections:
        print(
            f"    框{det.box_id}: ({det.x1:.1f}, {det.y1:.1f}) - "
            f"({det.x2:.1f}, {det.y2:.1f}), 置信度: {det.confidence:.3f}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    debug_output = output_path / f"debug_{image_idx:04d}_{rel_path.name}"
    draw_boxes(src_path, detections, debug_output)
    print(f"[*] 调试图片已保存: {debug_output}")


def write_failed_log(
    results: Sequence[ImageResult], output_path: Path
) -> Optional[Path]:
    """将失败结果写入日志文件。"""
    failed = [r for r in results if not r.success]
    if not failed:
        return None

    log_file = output_path / "_process_failed.log"
    with log_file.open("w", encoding="utf-8") as file:
        for result in failed:
            file.write(f"{result.relative_path}: {result.error}\n")
    return log_file


def preview_files(files: Sequence[Tuple[Path, Path]], limit: int = 5) -> bool:
    """显示待处理文件预览并请求确认。"""
    print("=" * 60)
    print(f"预览模式 (总计: {len(files)} 个)")
    print("=" * 60)

    for idx, (rel_path, _) in enumerate(files[:limit], start=1):
        print(f"  [{idx}] {rel_path}")

    if len(files) > limit:
        print(f"  ... 还有 {len(files) - limit} 个文件")

    print("\n" + "=" * 60)
    return input("确认执行? (y/n): ").strip().lower() == "y"


def main() -> None:
    settings = Settings()
    input_path = Path(path_win2wsl(settings.input_path))
    output_path = Path(path_win2wsl(settings.output_path))

    if not input_path.exists() or not input_path.is_dir():
        print(f"[-] 输入路径不存在或不是目录: {input_path}")
        return

    files = list(find_images(input_path))
    if not files:
        print("[-] 未找到图片文件")
        return

    print(f"[*] 找到 {len(files)} 个图片文件")
    print(f"\n[*] 加载模型: {settings.model_path}")

    try:
        model = YOLO(settings.model_path)
        print("[*] 模型加载成功")
    except Exception as exc:  # noqa: BLE001
        print(f"[-] 模型加载失败: {exc}")
        return

    if settings.debug_image_idx > 0:
        debug_image(
            settings.debug_image_idx,
            files,
            model,
            settings.conf_threshold,
            settings.end2end,
            output_path,
        )
        return

    if settings.preview and not preview_files(files):
        print("[!] 已取消")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    all_results: List[ImageResult] = []
    total_batches = (len(files) - 1) // settings.batch_size + 1
    print(f"\n[*] 开始处理 (batch_size={settings.batch_size})...")

    for start_idx in range(0, len(files), settings.batch_size):
        batch_idx = start_idx // settings.batch_size + 1
        batch = files[start_idx : start_idx + settings.batch_size]
        print(f"\n[Batch {batch_idx}/{total_batches}]")

        batch_results = process_batch(
            batch,
            model,
            settings.conf_threshold,
            settings.end2end,
        )
        all_results.extend(batch_results)

    csv_path = export_csv(all_results, output_path)
    print(f"\n[*] CSV结果已保存至: {csv_path}")

    ok_count = sum(result.success for result in all_results)
    fail_count = len(all_results) - ok_count
    print(f"\n完成: 成功 {ok_count}, 失败 {fail_count}")

    failed_log = write_failed_log(all_results, output_path)
    if failed_log:
        print(f"[!] 失败详情已记录至: {failed_log}")


if __name__ == "__main__":
    main()
