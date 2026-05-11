"""ESC 超分辨率推理脚本

用法:
    python infer/esc_infer.py
    python infer/esc_infer.py --input photo.jpg --output sr.png --fp16
"""

import time
from dataclasses import dataclass

import torch
import tyro
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel import ESC


@dataclass
class Config:
    """ESC x3 超分辨率推理"""
    input: str = "infer/images/debug_before_sr.png"
    output: str = "infer/images/debug_esc_x3.png"
    weight: str = "others/ESC_DFLIP_X3.pth"
    scale: int = 3
    fp16: bool = False
    attn: str = "Flex"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config: Config) -> ESC:
    model = ESC(
        dim=64, pdim=16, kernel_size=13,
        n_blocks=5, conv_blocks=5, window_size=32,
        num_heads=4, upscaling_factor=config.scale, exp_ratio=1.25,
        attn_type=config.attn,
    )
    ckpt = torch.load(config.weight, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt)

    if config.fp16:
        model = model.half()
        state_dict = {k: v.half() for k, v in state_dict.items()}

    model = model.to(config.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已加载: ESC x{config.scale} ({n_params:.1f}M) <- {config.weight}  (fp16={config.fp16}, attn={config.attn})")
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
    for _ in range(5):
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
