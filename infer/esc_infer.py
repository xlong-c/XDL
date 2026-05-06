"""ESC 超分辨率推理脚本 — 无 tiling 单次推理，直接跑即可。"""

import time

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel import ESC

# ═══════════════════════════════════════════════════════════
# 配置 
# ═══════════════════════════════════════════════════════════
INPUT = "infer/2_25.jpg"
OUTPUT = "infer/2_25_esc_x3.png"
SCALE = 3
FP16 = False
ATTN: str = "Flex"  # Naive | SDPA | Flex | FlashBias
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "dim": 64, "pdim": 16, "kernel_size": 13,
    "n_blocks": 5, "conv_blocks": 5, "window_size": 32,
    "num_heads": 4, "upscaling_factor": SCALE, "exp_ratio": 1.25,
}
WEIGHT = "others/ESC_DFLIP_X3.pth"


def load_model() -> ESC:
    model = ESC(**CFG, attn_type=ATTN)
    ckpt = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt)

    if FP16:
        model = model.half()
        state_dict = {k: v.half() for k, v in state_dict.items()}

    model = model.to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已加载: ESC x{SCALE} ({n_params:.1f}M) <- {WEIGHT}  (fp16={FP16}, attn={ATTN})")
    return model


def main():
    model = load_model()

    img = Image.open(INPUT).convert("RGB")
    print(f"输入: {INPUT} ({img.size[0]}x{img.size[1]})")

    dtype = torch.float16 if FP16 else torch.float32
    tensor = pil_to_tensor(img).to(dtype=dtype, device=DEVICE) / 255.0

    # warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(tensor.unsqueeze(0))
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # 推理计时
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(tensor.unsqueeze(0)).squeeze(0)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    out = out.clamp(0, 1)
    sr = to_pil_image(out.cpu())
    sr.save(OUTPUT)

    mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2 if DEVICE == "cuda" else 0
    print(f"输出: {OUTPUT} ({sr.size[0]}x{sr.size[1]})")
    print(f"推理耗时: {elapsed*1000:.1f}ms  |  峰值显存: {mem:.0f} MB")


if __name__ == "__main__":
    main()
