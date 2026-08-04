"""把 select_cloth 图片目录转成 SeFi dit_finetune 可读的本地 Parquet.

输出布局 (与 sefi.training.data.load_paired_rows 约定一致):

  <output_dir>/
    data/train-00000-of-00001.parquet
    manifest.json

每行字段:
  id, image(绝对路径), prompt, enhanced_prompt, caption

默认 caption 全部为 "outfit swap". 官方 FixedSquareImageTransform 会
resize+center-crop 到 1024, 不要求源图本来就是方图.

使用 (在 AutoDL 上):
  python build_select_cloth_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq

IMAGE_DIR = Path("/root/autodl-tmp/select_cloth")
OUTPUT_DIR = Path("/root/autodl-tmp/data/sefi_select_cloth")
CAPTION = "outfit swap"
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


def build_rows(paths: List[Path], caption: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        sample_id = f"{index:06d}_{path.stem}"
        rows.append(
            {
                "id": sample_id,
                "image": str(path),
                "prompt": caption,
                "enhanced_prompt": caption,
                "caption": caption,
            }
        )
    return rows


def write_parquet(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    if not rows:
        raise ValueError("没有可写入的样本")

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "train-00000-of-00001.parquet"

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)

    manifest = {
        "num_rows": len(rows),
        "caption": CAPTION,
        "image_dir": str(IMAGE_DIR),
        "parquet": str(parquet_path),
        "first_id": rows[0]["id"],
        "last_id": rows[-1]["id"],
        "first_image": rows[0]["image"],
        "last_image": rows[-1]["image"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return parquet_path


def main() -> None:
    paths = iter_image_paths(IMAGE_DIR, RECURSIVE)
    if not paths:
        raise ValueError(f"未找到图片: {IMAGE_DIR}")

    rows = build_rows(paths, CAPTION)
    parquet_path = write_parquet(rows, OUTPUT_DIR)
    print(f"records={len(rows)}")
    print(f"dataset={OUTPUT_DIR.resolve()}")
    print(f"parquet={parquet_path.resolve()}")
    print(f"first={rows[0]['image']}")
    print(f"last={rows[-1]['image']}")


if __name__ == "__main__":
    main()
