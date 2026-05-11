"""RealPLKSR 超分辨率推理脚本

RealPLKSR (ECCV 2024) — Partial Large Kernel CNNs for Efficient Super-Resolution.
用更大的 kernel (17x17) + 部分卷积 + Element-wise Attention 在效率与效果之间取得平衡。

用法:
    python infer/realplksr_infer.py
    python infer/realplksr_infer.py --input photo.jpg --output sr.png
"""

import time
from dataclasses import dataclass

import torch
import tyro
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel import realplksr


@dataclass
class Config:
    """RealPLKSR Nomos x4 超分辨率推理"""
    input: str = "infer/images/debug_before_sr.png"
    output: str = "infer/images/debug_realplksr_nomos_x4.png"
    weight: str = "others/4xNomosWebPhoto_RealPLKSR.safetensors"
    scale: int = 4
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config: Config):
    model = realplksr(
        dim=64,
        n_blocks=28,
        upscaling_factor=config.scale,
        kernel_size=17,
        split_ratio=0.25,
        use_ea=True,
        norm_groups=4,
        dropout=0,
    )
    state_dict = load_file(config.weight, device="cpu")

    if config.fp16:
        model = model.half()
        state_dict = {k: v.half() for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model = model.to(config.device).eval()

    if config.device == "cuda":
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model, mode="reduce-overhead")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已加载: RealPLKSR x{config.scale} ({n_params:.1f}M) <- {config.weight}  (fp16+compile)")
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
