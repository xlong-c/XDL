"""从图片目录生成 SeFi outfit LoRA 训练用 train.jsonl.

默认扫描 /root/autodl-tmp/select_cloth, 写出
/root/autodl-tmp/data/sefi_image_outfit/train.jsonl.
每行仅含 image 字段 (绝对路径), 与 sefi_image_outfit_lora.yaml 对齐.

使用:
  python examples/build_sefi_image_outfit_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

IMAGE_DIR = Path("/root/autodl-tmp/select_cloth")
OUTPUT_MANIFEST = Path("/root/autodl-tmp/data/sefi_image_outfit/train.jsonl")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
RECURSIVE = False


def iter_image_paths(image_dir: Path, recursive: bool) -> List[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    iterator: Iterable[Path]
    if recursive:
        iterator = image_dir.rglob("*")
    else:
        iterator = image_dir.iterdir()

    paths = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    return paths


def write_manifest(paths: List[Path], output_manifest: Path) -> None:
    if not paths:
        raise ValueError(f"未找到图片: {IMAGE_DIR}")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as file_obj:
        for path in paths:
            record = {"image": str(path)}
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    paths = iter_image_paths(IMAGE_DIR, RECURSIVE)
    write_manifest(paths, OUTPUT_MANIFEST)
    print(f"records={len(paths)}")
    print(f"manifest={OUTPUT_MANIFEST.resolve()}")
    print(f"first={paths[0]}")
    print(f"last={paths[-1]}")


if __name__ == "__main__":
    main()
