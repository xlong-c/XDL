"""RealPLKSR 超分辨率推理脚本

RealPLKSR (ECCV 2024) — Partial Large Kernel CNNs for Efficient Super-Resolution.
用更大的 kernel (17x17) + 部分卷积 + Element-wise Attention 在效率与效果之间取得平衡。
"""

import time

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel import realplksr

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
INPUT = "infer/images/3_2.jpg"
OUTPUT = "infer/images/3_2_realplksr_x4.png"
SCALE = 4
FP16 = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 从权重推导: dim=64, n_blocks=28, kernel_size=17, split_ratio=0.25
CFG = {
    "dim": 64,
    "n_blocks": 28,
    "upscaling_factor": SCALE,
    "kernel_size": 17,
    "split_ratio": 0.25,
    "use_ea": True,
    "norm_groups": 4,
    "dropout": 0,
}
WEIGHT = "others/4xNomosWebPhoto_RealPLKSR.safetensors"


def load_model() -> realplksr:
    model = realplksr(**CFG)
    state_dict = load_file(WEIGHT, device="cpu")

    if FP16:
        model = model.half()
        state_dict = {k: v.half() for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()

    torch.set_float32_matmul_precision("high")
    model = torch.compile(model, mode="reduce-overhead")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已加载: RealPLKSR x{SCALE} ({n_params:.1f}M) <- {WEIGHT}  (fp16+compile)")
    return model


def main():
    model = load_model()

    img = Image.open(INPUT).convert("RGB")
    w, h = img.size
    print(f"输入: {INPUT} ({w}x{h})")

    dtype = torch.float16 if FP16 else torch.float32
    tensor = pil_to_tensor(img).to(dtype=dtype, device=DEVICE) / 255.0

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
    sr.save(OUTPUT)

    mem = torch.cuda.max_memory_reserved(DEVICE) / 1024 ** 2 if DEVICE == "cuda" else 0
    print(f"输出: {OUTPUT} ({sr.size[0]}x{sr.size[1]})")
    print(f"推理耗时: {elapsed*1000:.1f}ms  |  显存占用: {mem:.0f} MB")


if __name__ == "__main__":
    main()
