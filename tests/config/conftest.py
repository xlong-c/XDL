"""为 examples/assets 下的示例清单补齐缺失样本图的测试夹具.

manifest jsonl 随仓库入库, 但 images/ 目录按数据规范不入库, 导致 CI 缺少
样本图. 本夹具在缺失时用 PIL 生成确定性的占位小图; 本地已有样本图时不做
任何修改.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "examples" / "assets"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _placeholder_image(path: Path) -> None:
    """生成确定性颜色的 16x16 占位图, 不覆盖已存在的文件."""
    digest = hashlib.sha256(path.stem.encode("utf-8")).digest()
    color = (digest[0], digest[1], digest[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def _collect_referenced_images(manifest_path: Path) -> list[Path]:
    """从 jsonl 清单里收集相对 manifest 目录的图片路径."""
    referenced: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for value in record.values():
                if not isinstance(value, str):
                    continue
                candidate = Path(value)
                if candidate.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                referenced.append(manifest_path.parent / candidate)
    return referenced


@pytest.fixture(scope="session", autouse=True)
def ensure_example_assets() -> None:
    """确保示例清单引用的样本图存在, 缺失时生成占位图."""
    for manifest_path in sorted(_ASSETS_DIR.glob("*/*.jsonl")):
        for image_path in _collect_referenced_images(manifest_path):
            if not image_path.exists():
                _placeholder_image(image_path)
