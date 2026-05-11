"""批量超分辨率模型对比

用同一张输入图跑所有可用模型，对比耗时和显存。

用法:
    python infer/batch_compare.py
    python infer/batch_compare.py --input photo.jpg
"""

import time
from dataclasses import dataclass

import torch
import tyro
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


@dataclass
class Config:
    """批量 SR 模型对比"""
    input: str = "infer/images/debug_before_sr.png"
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════
# 模型注册: (名称, scale, 加载函数)
# ═══════════════════════════════════════════════════════════

def load_aesop(config: Config):
    from xdl.model.lowlevel import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    ckpt = torch.load("others/RealAESOP_RRDB256_400K.pth", map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema", ckpt)
    if config.fp16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(config.device).eval()

def load_esc(config: Config):
    from xdl.model.lowlevel import ESC
    model = ESC(dim=64, pdim=16, kernel_size=13, n_blocks=5, conv_blocks=5, window_size=32, num_heads=4, upscaling_factor=3, exp_ratio=1.25)
    ckpt = torch.load("others/ESC_DFLIP_X3.pth", map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema", ckpt)
    if config.fp16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(config.device).eval()

def load_realesrgan(config: Config):
    from xdl.model.lowlevel import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    ckpt = torch.load("others/RealESRGAN_x2plus.pth", map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema", ckpt)
    if config.fp16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(config.device).eval()

def load_realplksr_nomos(config: Config):
    from safetensors.torch import load_file
    from xdl.model.lowlevel import realplksr
    model = realplksr(dim=64, n_blocks=28, upscaling_factor=4, kernel_size=17, split_ratio=0.25, use_ea=True, norm_groups=4, dropout=0)
    sd = load_file("others/4xNomosWebPhoto_RealPLKSR.safetensors", device="cpu")
    if config.fp16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(config.device).eval()

def load_realplksr_ult(config: Config):
    from xdl.model.lowlevel.realplksr_arch_ult import realplksr as realplksr_ult
    model = realplksr_ult(dim=64, n_blocks=28, scale=4, kernel_size=17, split_ratio=0.25, use_ea=True, norm_groups=4, dropout=0, upsampler="pixelshuffle", layer_norm=True)
    ckpt = torch.load("others/4x-UltraSharpV2_Lite.pth", map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema", ckpt.get("params", ckpt))
    if config.fp16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(config.device).eval()

MODELS = [
    ("AESOP x4",       4, load_aesop),
    ("ESC x3",         3, load_esc),
    ("RealESRGAN x2",  2, load_realesrgan),
    ("RealPLKSR Nomos x4", 4, load_realplksr_nomos),
    ("RealPLKSR Ultra x4", 4, load_realplksr_ult),
]


def main():
    config = tyro.cli(Config)

    torch.set_float32_matmul_precision("high")

    img = Image.open(config.input).convert("RGB")
    w, h = img.size
    print(f"输入: {config.input} ({w}x{h})\n")

    results = []
    for name, scale, loader in MODELS:
        print(f"--- {name} ---")
        try:
            model = loader(config)
        except Exception as e:
            print(f"  加载失败: {e}\n")
            continue

        n_params = sum(p.numel() for p in model.parameters()) / 1e6

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
        out_path = f"infer/images/debug_batch_{name.replace(' ', '_').lower()}.png"
        sr.save(out_path)

        mem = torch.cuda.max_memory_reserved(config.device) / 1024**2 if config.device == "cuda" else 0
        print(f"  {n_params:.1f}M  |  {elapsed*1000:.1f}ms  |  {mem:.0f} MB  |  {sr.size[0]}x{sr.size[1]}  ->  {out_path}")

        results.append((name, n_params, elapsed * 1000, mem, sr.size))
        del model, tensor, out

    if not results:
        print("没有成功运行的模型。")
        return

    print(f"\n{'='*80}")
    print(f"{'模型':<22s} {'参数(M)':>8s} {'耗时(ms)':>10s} {'显存(MB)':>10s} {'输出尺寸':>12s}")
    print(f"{'-'*80}")
    for name, params, ms, mem, size in results:
        print(f"{name:<22s} {params:>8.1f} {ms:>10.1f} {mem:>10.0f} {size[0]:>5d}x{size[1]:<5d}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
