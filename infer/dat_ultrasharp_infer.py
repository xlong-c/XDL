"""DAT (Dual Aggregation Transformer) UltraSharpV2 超分辨率推理脚本

使用 DAT 架构 (dat_2 配置).
对应权重: 4x-UltraSharpV2.pth (Kim2091)
"""

import time
from dataclasses import dataclass

import torch
import tyro
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel.dat_arch import dat_2
from xdl.utils.tiling import tile_inference


@dataclass
class Config:
    """DAT UltraSharpV2 推理配置"""
    input: str = "infer/images/debug_before_sr.png"
    output: str = "infer/images/debug_dat_ultrasharp_x4.png"
    weight: str = "others/4x-UltraSharpV2.pth"
    scale: int = 4
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tile: bool = True
    tile_size: int = 64
    tile_pad: int = 8


def load_model(config: Config):
    model = dat_2(
        scale=config.scale,
        in_chans=3,
        img_size=64,
        img_range=1.0,
        split_size=(8, 32),
        depth=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6),
        expansion_factor=2,
        resi_connection="1conv",
        upsampler="pixelshuffle",
    )

    ckpt = torch.load(config.weight, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))

    if config.fp16:
        model = model.half()
        state_dict = {k: v.half() for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model = model.to(config.device).eval()

    if config.device == "cuda":
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model, mode="reduce-overhead")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已加载: DAT (dat_2) x{config.scale} ({n_params:.1f}M) <- {config.weight}  (fp16+compile)")
    return model


def main():
    config = tyro.cli(Config)

    model = load_model(config)

    img = Image.open(config.input).convert("RGB")
    w, h = img.size
    print(f"输入: {config.input} ({w}x{h})")

    dtype = torch.float16 if config.fp16 else torch.float32
    tensor = pil_to_tensor(img).to(dtype=dtype, device=config.device) / 255.0

    # warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(tensor.unsqueeze(0))
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        if config.tile:
            out = tile_inference(model, tensor, tile_size=config.tile_size, tile_pad=config.tile_pad, scale=config.scale)
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
