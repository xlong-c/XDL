"""WFEN 人脸超分辨率推理脚本

WFEN (ACM MM 2024) — 小波 + 全领域 Transformer 人脸增强网络。
先 bicubic 上采样到目标尺寸，再分块 WFEN 增强。

用法:
    python infer/wfen_infer.py
    python infer/wfen_infer.py --input photo.jpg --output sr.png --tile-size 128
"""

import time
from dataclasses import dataclass

import torch
import tyro
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel.wfen_arch import WFEN
from xdl.utils.tiling import tile_inference


@dataclass
class Config:
    """WFEN x4 人脸超分辨率推理"""
    input: str = "infer/images/debug_before_sr.png"
    output: str = "infer/images/debug_wfen_x4.png"
    weight: str = "others/WFEN.pth"
    scale: int = 4
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tile: bool = True
    tile_size: int = 128
    tile_pad: int = 16


def load_model(config: Config) -> WFEN:
    model = WFEN(inchannel=3, min_ch=40, res_depth=6)
    sd = torch.load(config.weight, map_location="cpu", weights_only=True)

    if config.fp16:
        model = model.half()
        sd = {k: v.half() for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    model = model.to(config.device).eval()

    if config.device == "cuda":
        torch.set_float32_matmul_precision("high")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已加载: WFEN ({n_params:.1f}M) <- {config.weight}  (fp16={config.fp16})")
    return model


def main():
    config = tyro.cli(Config)

    model = load_model(config)

    img = Image.open(config.input).convert("RGB")
    w, h = img.size
    print(f"输入: {config.input} ({w}x{h})")

    # bicubic 上采样到目标尺寸
    img_hr = img.resize((w * config.scale, h * config.scale), Image.BICUBIC)

    dtype = torch.float16 if config.fp16 else torch.float32
    tensor = pil_to_tensor(img_hr).to(dtype=dtype, device=config.device) / 255.0

    # warmup
    test_tile = tensor[:, :config.tile_size, :config.tile_size].unsqueeze(0)
    for _ in range(2):
        with torch.no_grad():
            _ = model(test_tile)
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        if config.tile:
            out = tile_inference(model, tensor, tile_size=config.tile_size, tile_pad=config.tile_pad, scale=1)
        else:
            out = model(tensor.unsqueeze(0)).squeeze(0)
    if config.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    out = out.clamp(0, 1)
    sr = to_pil_image(out.cpu())
    sr.save(config.output)

    mem = torch.cuda.max_memory_reserved(config.device) / 1024 ** 2 if config.device == "cuda" else 0
    print(f"输出: {config.output} ({sr.size[0]}x{sr.size[1]})")
    print(f"推理耗时: {elapsed*1000:.1f}ms  |  显存占用: {mem:.0f} MB")


if __name__ == "__main__":
    main()
