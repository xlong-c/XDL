"""AESOP 超分辨率推理脚本

AESOP (CVPR 2025) 是一种训练方法创新 — 用预训练 AutoEncoder 的
输出空间替代像素空间计算 fidelity loss。推理时 SR 网络本身仍是 RRDBNet。
"""

import time

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel import RRDBNet

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
INPUT = "infer/images/3_2.jpg"
OUTPUT = "infer/images/3_2_aesop_x4.png"
SCALE = 4
FP16 = True  # FP16 + torch.compile: 显存 139MB, 1.7x 加速 vs 纯 FP16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "num_in_ch": 3, "num_out_ch": 3,
    "num_feat": 64, "num_block": 23, "num_grow_ch": 32,
    "scale": SCALE,
}
WEIGHT = "others/RealAESOP_RRDB256_400K.pth"


def load_model() -> RRDBNet:
    model = RRDBNet(**CFG)
    ckpt = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params_ema", ckpt)

    if FP16:
        model = model.half()
        state_dict = {k: v.half() for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()

    torch.set_float32_matmul_precision("high")
    model = torch.compile(model, mode="reduce-overhead")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已加载: AESOP-SR x{SCALE} ({n_params:.1f}M) <- {WEIGHT}  (fp16+compile)")
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
