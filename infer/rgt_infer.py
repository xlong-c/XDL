"""RGT 超分辨率推理脚本。

用法:
    python infer/rgt_infer.py -i photo.jpg -o sr.png            # RGT (8层, 默认)
    python infer/rgt_infer.py -i photo.jpg -o sr.png --model rgt_s  # RGT_S (6层, 更快)
    python infer/rgt_infer.py -i photo.jpg --fp16 --batch 8     # FP16 加速
    python infer/rgt_infer.py -i photo.jpg --tile 0             # 不分块 (需大显存)

模型:
    rgt   (默认): 8层, 13.2M, ~1.5s, others/RGT_x2.pth
    rgt_s:        6层, 10.1M, ~1.2s, others/RGT_S_x2.pth
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel import RGT
from xdl.utils.tiling import tile_inference

# 预置模型配置
PRESETS = {
    "rgt": {
        "weight": "others/RGT_x2.pth",
        "cfg": {
            "img_size": 64, "in_chans": 3, "embed_dim": 180,
            "depth": [6, 6, 6, 6, 6, 6, 6, 6],
            "num_heads": [6, 6, 6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2, "upscale": 2, "split_size": [8, 32],
            "c_ratio": 0.5, "resi_connection": "1conv", "img_range": 1.0,
        },
    },
    "rgt_s": {
        "weight": "others/RGT_S_x2.pth",
        "cfg": {
            "img_size": 64, "in_chans": 3, "embed_dim": 180,
            "depth": [6, 6, 6, 6, 6, 6],
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2, "upscale": 2, "split_size": [8, 32],
            "c_ratio": 0.5, "resi_connection": "1conv", "img_range": 1.0,
        },
    },
}


def load_model(model_name: str, device: str = "cuda", fp16: bool = False) -> RGT:
    preset = PRESETS[model_name]
    model = RGT(**preset["cfg"])
    ckpt = torch.load(preset["weight"], map_location="cpu", weights_only=True)
    state_dict = ckpt["params"]

    if fp16:
        model = model.half()
        state_dict = {k: v.half() for k, v in state_dict.items()}

    model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已加载: {model_name} ({n_params:.1f}M) <- {preset['weight']}  (fp16={fp16})")
    return model


def infer_image(
    model: RGT,
    image: Image.Image,
    tile_size: int = 64,
    tile_pad: int = 8,
    batch_size: int = 8,
    fp16: bool = False,
    device: str = "cuda",
) -> Image.Image:
    dtype = torch.float16 if fp16 else torch.float32
    img_tensor = pil_to_tensor(image).to(dtype=dtype, device=device) / 255.0

    if tile_size > 0:
        sr_tensor = tile_inference(
            model, img_tensor, tile_size=tile_size, tile_pad=tile_pad,
            scale=2, batch_size=batch_size,
        )
    else:
        with torch.no_grad():
            sr_tensor = model(img_tensor.unsqueeze(0)).squeeze(0)

    sr_tensor = sr_tensor.clamp(0, 1)
    return to_pil_image(sr_tensor.cpu())


def main():
    parser = argparse.ArgumentParser(description="RGT x2 超分辨率推理")
    parser.add_argument("-i", "--input", required=True, help="输入图像路径")
    parser.add_argument("-o", "--output", default="sr_output.png", help="输出图像路径")
    parser.add_argument("--model", default="rgt", choices=["rgt", "rgt_s"], help="模型选择")
    parser.add_argument("--tile", type=int, default=64, help="分块大小 (0=不分块)")
    parser.add_argument("--pad", type=int, default=8, help="分块重叠宽度")
    parser.add_argument("--batch", type=int, default=8, help="tile 批量大小")
    parser.add_argument("--fp16", action="store_true", help="FP16 半精度推理")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_path = Path(args.input)

    print(f"设备: {args.device}  |  {args.model}  |  fp16={args.fp16}  |  tile={args.tile}  |  batch={args.batch}")
    print(f"输入: {input_path}")

    model = load_model(args.model, device=args.device, fp16=args.fp16)

    image = Image.open(input_path).convert("RGB")
    print(f"图像尺寸: {image.size[0]}x{image.size[1]}")

    t0 = time.perf_counter()
    sr_image = infer_image(
        model, image, tile_size=args.tile, tile_pad=args.pad,
        batch_size=args.batch, fp16=args.fp16, device=args.device,
    )
    elapsed = time.perf_counter() - t0

    print(f"输出尺寸: {sr_image.size[0]}x{sr_image.size[1]}")
    print(f"推理耗时: {elapsed:.2f}s")
    sr_image.save(args.output)
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
