"""ATD 超分辨率推理脚本。

用法:
    python infer/atd_infer.py
    python infer/atd_infer.py --input photo.jpg --output sr.png
    python infer/atd_infer.py --input photo.jpg --fp16 --tile-size 96
"""

import time
from dataclasses import dataclass

import torch
import tyro
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel import ATD
from xdl.utils.tiling import tile_inference

# ATD x2 模型架构（固定参数）
MODEL_CFG = {
    "img_size": 96,
    "in_chans": 3,
    "embed_dim": 216,
    "depths": [6, 6, 6, 6, 6, 6],
    "num_heads": [4, 4, 4, 4, 4, 4],
    "window_size": 16,
    "dim_ffn_td": 16,
    "category_size": 256,
    "num_tokens": 512,
    "reducted_dim": 16,
    "convffn_kernel_size": 5,
    "mlp_ratio": 2,
    "upscale": 2,
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
    "img_range": 1.0,
}


@dataclass
class Config:
    """ATD x2 超分辨率推理"""
    input: str = "infer/images/debug_before_sr.png"
    output: str = "infer/images/debug_atd_x2.png"
    weight: str = "others/001_ATD_SRx2_finetune.pth"
    fp16: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tile: bool = True
    tile_size: int = 96
    tile_pad: int = 12
    batch_size: int = 4


def load_model(config: Config) -> ATD:
    model = ATD(**MODEL_CFG)
    ckpt = torch.load(config.weight, map_location="cpu", weights_only=True)
    state_dict = ckpt["params"]

    if config.fp16:
        model = model.half()
        state_dict = {k: v.half() for k, v in state_dict.items()}

    model = model.to(config.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已加载: ATD x2 ({n_params:.1f}M) <- {config.weight}  (fp16={config.fp16})")
    return model


def main():
    config = tyro.cli(Config)

    print(f"设备: {config.device}  |  fp16={config.fp16}  |  tile={config.tile_size if config.tile else 0}  |  batch={config.batch_size}")

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
            out = tile_inference(model, tensor, tile_size=config.tile_size, tile_pad=config.tile_pad, scale=2, batch_size=config.batch_size)
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
