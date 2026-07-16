"""Quantize tencent/HunyuanOCR with SVDQuant FP4 storage and INT8 MMA inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt.model.hunyuan_ocr import (
    HUNYUAN_OCR_REPO_ID,
    load_hunyuan_ocr,
    optimize_hunyuan_ocr_svd_fp4_int8_mma,
)


CONFIG: dict[str, Any] = {
    "model_id": HUNYUAN_OCR_REPO_ID,
    "revision": None,
    "local_files_only": False,
    "dtype": torch.bfloat16,
    "artifact_dir": "artifacts/xqt/hunyuan_ocr_svd_fp4_int8_mma",
    "quantization": {
        "rank": 32,
        "group_size": 128,
        "engine": "auto",
        "fallback_engine": "torch_int_mm",
        "policy": {
            "include_module_types": ["Linear"],
            "exclude_name_patterns": [],
            "activation_scale_mode": "dynamic",
        },
    },
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    artifact_dir = Path(str(CONFIG["artifact_dir"]))
    quantization = dict(CONFIG["quantization"])
    model = load_hunyuan_ocr(
        repo_id=str(CONFIG["model_id"]),
        revision=CONFIG["revision"],
        dtype=CONFIG["dtype"],
        device=_device(),
        local_files_only=bool(CONFIG["local_files_only"]),
    )
    result = optimize_hunyuan_ocr_svd_fp4_int8_mma(
        model,
        artifact_dir=artifact_dir,
        rank=int(quantization["rank"]),
        group_size=int(quantization["group_size"]),
        engine=str(quantization["engine"]),
        fallback_engine=str(quantization["fallback_engine"]),
        policy=quantization["policy"],
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "hunyuan_ocr_svd_fp4_int8_mma.json"
    report_path.write_text(
        json.dumps(
            {
                "model_id": CONFIG["model_id"],
                "device": str(_device()),
                "quant_stage": result.stage.metrics,
                "compute_config": result.compute_config,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
