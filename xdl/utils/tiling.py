"""分块推理（Tiling）工具 —— 用小块训练模型对大图做超分推理。"""

import torch
from torch import Tensor


def _blend_mask(h: int, w: int, pad: int) -> Tensor:
    """生成从边缘到中心的渐变权重 mask，用于平滑拼接 tile 边界。

    边缘处权重趋近 0，中心区域权重为 1，过渡区使用 sin 曲线。
    """
    y = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
    x = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)

    def _ramp(pos: Tensor, max_pos: int) -> Tensor:
        ramp_size = min(pad, max_pos // 2)
        mask = torch.ones_like(pos)
        if ramp_size > 0:
            mask = torch.minimum(mask, torch.sin(torch.clamp(pos / ramp_size, 0, 1) * 1.5707963))
            mask = torch.minimum(
                mask, torch.sin(torch.clamp((max_pos - 1 - pos) / ramp_size, 0, 1) * 1.5707963)
            )
        return mask

    return _ramp(y, h) * _ramp(x, w)


def tile_inference(
    model: torch.nn.Module,
    image: Tensor,
    tile_size: int = 64,
    tile_pad: int = 8,
    scale: int = 2,
    batch_size: int = 1,
) -> Tensor:
    """分块推理：将大图切成重叠 tile 分别超分后拼接回来。

    Args:
        model: 超分模型，forward 接受 (B, C, H, W)，返回 (B, C, H*scale, W*scale)。
        image:  输入图像 (C, H, W) 或 (B, C, H, W)。
        tile_size: 每个 tile 的输入尺寸（默认为模型训练尺寸）。
        tile_pad:  tile 重叠宽度（防止接缝），建议 >= tile_size // 8。
        scale:    超分倍率。
        batch_size: 一次喂给模型的 tile 数量。

    Returns:
        与 image 同格式的超分结果。
    """
    squeeze_batch = image.dim() == 3
    if squeeze_batch:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape
    stride = tile_size - 2 * tile_pad
    if stride <= 0:
        raise ValueError(f"tile_pad ({tile_pad}) must be less than tile_size/2 ({tile_size / 2})")

    # 计算 tile 网格
    y_steps = max(1, (H - 2 * tile_pad + stride - 1) // stride)
    x_steps = max(1, (W - 2 * tile_pad + stride - 1) // stride)
    y_stride = max(1, (H - 2 * tile_pad + y_steps - 1) // y_steps)
    x_stride = max(1, (W - 2 * tile_pad + x_steps - 1) // x_steps)

    out_H, out_W = H * scale, W * scale
    out_tile_h = tile_size * scale
    out_tile_w = tile_size * scale
    out_pad = tile_pad * scale

    device = image.device
    output = torch.zeros(B, C, out_H, out_W, dtype=torch.float32, device=device)
    weight = torch.zeros(B, 1, out_H, out_W, dtype=torch.float32, device=device)

    blend = _blend_mask(out_tile_h, out_tile_w, out_pad).to(device=device, dtype=torch.float32)

    # 收集所有 tile 坐标：(y_in, x_in, y_out, x_out)
    tiles: list[tuple[int, int, int, int]] = []
    for yi in range(y_steps):
        y_center = tile_pad + yi * y_stride
        for xi in range(x_steps):
            x_center = tile_pad + xi * x_stride
            x_in = max(0, min(x_center - tile_pad, W - tile_size))
            y_in = max(0, min(y_center - tile_pad, H - tile_size))
            tiles.append((y_in, x_in, y_in * scale, x_in * scale))

    # 批处理
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i : i + batch_size]
        tile_batch: list[Tensor] = []
        for y_in, x_in, _, _ in batch_tiles:
            tile_batch.append(image[0, :, y_in : y_in + tile_size, x_in : x_in + tile_size])

        inp = torch.stack(tile_batch, dim=0).to(device)
        with torch.no_grad():
            out_tiles: Tensor = model(inp)

        for j, (_, _, y_out, x_out) in enumerate(batch_tiles):
            output[0, :, y_out : y_out + out_tile_h, x_out : x_out + out_tile_w] += out_tiles[j] * blend
            weight[0, :, y_out : y_out + out_tile_h, x_out : x_out + out_tile_w] += blend

    output = output / weight.clamp_min(1e-8)

    if squeeze_batch:
        output = output.squeeze(0)
    return output
