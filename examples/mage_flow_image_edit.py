"""Mage-Flow 指令式图像编辑示例.

使用前先按官方仓库安装 Mage-Flow 及其 CUDA 依赖:

    git clone https://github.com/microsoft/Mage.git
    cd Mage/mage_flow
    pip install -r requirements.txt
    pip install -e . --no-deps
    pip install --no-build-isolation flash-attn==2.8.3

本示例将 ``others/pics/p1.webp`` 中人物的服装替换为
``others/pics/p2.webp`` 的蓝白古装. 直接运行:

    python examples/mage_flow_image_edit.py

首次运行会下载 ``microsoft/Mage-Flow-Edit-Turbo`` 权重. ``Mage-Flow`` 本身
是文生图 checkpoint; 指令编辑必须使用 ``Mage-Flow-Edit`` 系列权重.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final


CONFIG: Final[dict[str, Any]] = {
    "model_id": "microsoft/Mage-Flow-Edit-Turbo",
    "device": "cuda",
    "reference_images": [
        "others/pics/p1.webp",
        "others/pics/p2.webp",
    ],
    "output_image": "others/pics/p1_wearing_p2_clothes.png",
    "prompt": (
        "Use image 1 as the identity and scene reference. Keep its person, face, "
        "hairstyle, expression, pose, body proportions, hands, sword, outdoor garden "
        "background, and lighting unchanged. Use image 2 only as the clothing reference. "
        "Replace image 1's white hanfu with image 2's blue-and-white traditional Chinese "
        "hanfu: a deep blue embroidered outer robe with wide flowing sleeves, a white "
        "inner robe and long white skirt, pale-blue gauze panels, silver-white floral "
        "embroidery, a blue waist sash, and long blue ribbons. Do not add the birdcage "
        "or any other prop from image 2. Do not add people, text, or watermarks."
    ),
    "height": 768,
    "width": 768,
    "steps": 4,
    "cfg": 1.0,
    "seed": 42,
    "vl_cond_long_edge": 384,
}


def _validate_config(config: dict[str, Any]) -> tuple[list[Path], Path]:
    device = str(config["device"])
    if not device.startswith("cuda"):
        raise ValueError("Mage-Flow-Edit 示例仅支持 CUDA 设备")

    reference_images = [
        Path(str(raw_path)) for raw_path in list(config["reference_images"])
    ]
    if len(reference_images) != 2:
        raise ValueError("服装迁移需要恰好两张参考图: p1 人物图与 p2 服装图")
    for reference_image in reference_images:
        if not reference_image.is_file():
            raise FileNotFoundError(f"找不到参考图片: {reference_image}")

    height = int(config["height"])
    width = int(config["width"])
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError("`height` 和 `width` 必须是 16 的倍数")
    if not 512 <= height <= 2048 or not 512 <= width <= 2048:
        raise ValueError("Mage-Flow 原生分辨率范围为 512 到 2048")

    output_image = Path(str(config["output_image"]))
    output_image.parent.mkdir(parents=True, exist_ok=True)
    return reference_images, output_image


def _load_pipeline(model_id: str, device: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Mage-Flow-Edit 需要安装 PyTorch CUDA 版本") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA 设备; Mage-Flow-Edit 需要 CUDA")

    try:
        from mage_flow import MageFlowPipeline
    except ImportError as exc:
        raise RuntimeError(
            "未安装 mage-flow. 请按本文件顶部的官方安装步骤安装 Mage-Flow."
        ) from exc

    return MageFlowPipeline.from_pretrained(model_id, device=device)


def main() -> None:
    reference_images, output_image = _validate_config(CONFIG)
    pipeline = _load_pipeline(str(CONFIG["model_id"]), str(CONFIG["device"]))
    images = pipeline.edit(
        [str(CONFIG["prompt"])],
        [[str(reference_image) for reference_image in reference_images]],
        heights=[int(CONFIG["height"])],
        widths=[int(CONFIG["width"])],
        steps=int(CONFIG["steps"]),
        cfg=float(CONFIG["cfg"]),
        seeds=[int(CONFIG["seed"])],
        vl_cond_long_edge=int(CONFIG["vl_cond_long_edge"]),
    )
    if len(images) != 1:
        raise RuntimeError(f"预期得到 1 张编辑结果, 实际得到 {len(images)} 张")
    images[0].save(output_image)
    print(f"编辑结果已保存到: {output_image}")


if __name__ == "__main__":
    main()
