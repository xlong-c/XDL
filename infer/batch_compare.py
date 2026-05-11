"""批量超分辨率模型对比

用同一张输入图跑所有可用模型，对比耗时和显存。
"""

import time
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True
INPUT = "infer/images/meee_512.jpg"

# ═══════════════════════════════════════════════════════════
# 模型注册: (名称, scale, 加载函数)
# ═══════════════════════════════════════════════════════════

def load_aesop():
    from xdl.model.lowlevel import RRDBNet
    cfg = {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 4}
    model = RRDBNet(**cfg)
    ckpt = torch.load("others/RealAESOP_RRDB256_400K.pth", map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema", ckpt)
    if FP16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()

def load_esc():
    from xdl.model.lowlevel import ESC
    model = ESC(dim=64, pdim=16, kernel_size=13, n_blocks=5, conv_blocks=5, window_size=32, num_heads=4, upscaling_factor=3, exp_ratio=1.25)
    ckpt = torch.load("others/ESC_DFLIP_X3.pth", map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema", ckpt)
    if FP16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()

def load_realesrgan():
    from xdl.model.lowlevel import RRDBNet
    cfg = {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 2}
    model = RRDBNet(**cfg)
    ckpt = torch.load("others/RealESRGAN_x2plus.pth", map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema", ckpt)
    if FP16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()

def load_realplksr_nomos():
    from safetensors.torch import load_file
    from xdl.model.lowlevel import realplksr
    model = realplksr(dim=64, n_blocks=28, upscaling_factor=4, kernel_size=17, split_ratio=0.25, use_ea=True, norm_groups=4, dropout=0)
    sd = load_file("others/4xNomosWebPhoto_RealPLKSR.safetensors", device="cpu")
    if FP16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()

def load_realplksr_ult():
    from xdl.model.lowlevel.realplksr_arch_ult import realplksr as realplksr_ult
    model = realplksr_ult(dim=64, n_blocks=28, scale=4, kernel_size=17, split_ratio=0.25, use_ea=True, norm_groups=4, dropout=0, upsampler="pixelshuffle", layer_norm=True)
    ckpt = torch.load("others/4x-UltraSharpV2_Lite.pth", map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema", ckpt.get("params", ckpt))
    if FP16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()

def load_atd():
    from xdl.model.lowlevel import ATD
    model = ATD(img_size=96, in_chans=3, embed_dim=216, depths=[6, 6, 6, 6, 6, 6], num_heads=[4, 4, 4, 4, 4, 4], window_size=16, dim_ffn_td=16, category_size=256, num_tokens=512, upscale=2, qkv_bias=True, global_attn=False, resi_connection="1conv")
    ckpt = torch.load("others/001_ATD_SRx2_finetune.pth", map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema", ckpt)
    if FP16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()

def load_rgt():
    from xdl.model.lowlevel import RGT
    model = RGT(img_size=64, in_chans=3, embed_dim=180, depth=[6, 6, 6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6, 6, 6], mlp_ratio=2, upscale=2, split_size=[8, 32], c_ratio=0.5, resi_connection="1conv", img_range=1.0)
    ckpt = torch.load("others/RGT_x2.pth", map_location="cpu", weights_only=True)
    sd = ckpt["params"]
    if FP16:
        model, sd = model.half(), {k: v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()

MODELS = [
    ("AESOP x4",       4,  load_aesop),
    ("ESC x3",         3,  load_esc),
    ("RealESRGAN x2",  2,  load_realesrgan),
    ("RealPLKSR Nomos x4", 4, load_realplksr_nomos),
    ("RealPLKSR Ultra x4", 4, load_realplksr_ult),
]


def main():
    torch.set_float32_matmul_precision("high")

    img = Image.open(INPUT).convert("RGB")
    w, h = img.size
    print(f"输入: {INPUT} ({w}x{h})\n")

    results = []
    for name, scale, loader in MODELS:
        print(f"--- {name} ---")
        try:
            model = loader()
        except Exception as e:
            print(f"  加载失败: {e}\n")
            continue

        # model = torch.compile(model, mode="reduce-overhead")  # FP16 inductor bmm 有兼容性问题
        n_params = sum(p.numel() for p in model.parameters()) / 1e6

        dtype = torch.float16 if FP16 else torch.float32
        tensor = pil_to_tensor(img).to(dtype=dtype, device=DEVICE) / 255.0

        # warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(tensor.unsqueeze(0))
        if DEVICE == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(tensor.unsqueeze(0)).squeeze(0)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        out = out.clamp(0, 1)
        sr = to_pil_image(out.cpu())
        out_path = f"infer/images/meee_512_{name.replace(' ', '_').lower()}.png"
        sr.save(out_path)

        mem = torch.cuda.max_memory_reserved(DEVICE) / 1024**2 if DEVICE == "cuda" else 0
        print(f"  {n_params:.1f}M  |  {elapsed*1000:.1f}ms  |  {mem:.0f} MB  |  {sr.size[0]}x{sr.size[1]}  ->  {out_path}")

        results.append((name, n_params, elapsed * 1000, mem, sr.size))
        del model, tensor, out

    # 汇总
    print(f"\n{'='*80}")
    print(f"{'模型':<22s} {'参数(M)':>8s} {'耗时(ms)':>10s} {'显存(MB)':>10s} {'输出尺寸':>12s}")
    print(f"{'-'*80}")
    for name, params, ms, mem, size in results:
        print(f"{name:<22s} {params:>8.1f} {ms:>10.1f} {mem:>10.0f} {size[0]:>5d}x{size[1]:<5d}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
