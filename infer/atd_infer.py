"""ATD 超分辨率推理脚本。

用法:
    python infer/atd_infer.py -i photo.jpg -o sr_photo.png
    python infer/atd_infer.py -i photo.jpg -o sr.png --fp16 --batch 4

默认从 others/001_ATD_SRx2_finetune.pth 加载 x2 权重。
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from xdl.model.lowlevel import ATD
from xdl.utils.tiling import tile_inference

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


def load_atd(weight_path: str, device: str = "cuda", fp16: bool = False) -> ATD:
    """加载 ATD 模型并载入预训练权重。"""
    model = ATD(**MODEL_CFG)
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["params"]

    if fp16:
        model = model.half()
        state_dict = {k: v.half() for k, v in state_dict.items()}

    model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"已加载权重: {weight_path}  (fp16={fp16})")
    return model


def infer_image(
    model: ATD,
    image: Image.Image,
    tile_size: int = 96,
    tile_pad: int = 12,
    batch_size: int = 4,
    fp16: bool = False,
    device: str = "cuda",
) -> Image.Image:
    """对单张 PIL 图像做超分推理。"""
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
    parser = argparse.ArgumentParser(description="ATD x2 超分辨率推理")
    parser.add_argument("-i", "--input", required=True, help="输入图像路径")
    parser.add_argument("-o", "--output", default="sr_atd_output.png", help="输出图像路径")
    parser.add_argument("--weight", default="others/001_ATD_SRx2_finetune.pth", help="预训练权重路径")
    parser.add_argument("--tile", type=int, default=96, help="分块大小 (0=不分块)")
    parser.add_argument("--pad", type=int, default=12, help="分块重叠宽度")
    parser.add_argument("--batch", type=int, default=4, help="tile 批量大小")
    parser.add_argument("--fp16", action="store_true", help="FP16 半精度推理")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    weight_path = root / args.weight if not Path(args.weight).is_absolute() else Path(args.weight)
    input_path = Path(args.input)

    print(f"设备: {args.device}  |  fp16={args.fp16}  |  tile={args.tile}  |  batch={args.batch}")
    print(f"输入: {input_path}")

    model = load_atd(str(weight_path), device=args.device, fp16=args.fp16)

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
