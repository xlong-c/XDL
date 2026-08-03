"""Unlimited-OCR 推理薄入口.

用法:
    python infer/unlimited_ocr_infer.py
    python infer/unlimited_ocr_infer.py research/unlimited-ocr/configs/transformers_single_image.yaml
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "unlimited-ocr"
        / "run_transformers_infer.py"
    )
    spec = importlib.util.spec_from_file_location("research_unlimited_ocr_infer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load research script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
