"""OFTSR Guided UNet — 条件 Flow Matching 超分辨率网络

来源: OFTSR (ICLR 2026), yuanzhi-zhu/OFTSR
许可: 未标注（公开仓库）

基于 OpenAI Guided Diffusion UNet，适配为条件 Flow Matching 教师/学生模型。
支持 timestep 条件、类别条件、低分辨率图像条件。
"""

from abc import abstractmethod
import math
from typing import Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 神经网络工具 (从 models/nn.py 内联)
# ---------------------------------------------------------------------------

class SiLU(nn.Module):
    """PyTorch 1.5 兼容的 SiLU 激活。"""
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x)


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs) -> nn.Module:
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels: int) -> nn.Module:
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps: th.Tensor, dim: int, max_period: int = 10000) -> th.Tensor:
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period)
        * th.arange(start=0, end=half, dtype=th.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None] * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag: bool):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors, ctx.input_params, output_tensors
        return (None, None) + input_grads


# ---------------------------------------------------------------------------
# fp16/fp32 转换
# ---------------------------------------------------------------------------

def convert_module_to_f16(module: nn.Module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.half()
        if module.bias is not None:
            module.bias.data = module.bias.data.half()


def convert_module_to_f32(module: nn.Module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.float()
        if module.bias is not None:
            module.bias.data = module.bias.data.float()


# ---------------------------------------------------------------------------
# 注意力
# ---------------------------------------------------------------------------

class QKVAttention(nn.Module):
    """QKV 注意力 — head split 在 chunk 之前。"""
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight, dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class QKVAttentionLegacy(nn.Module):
    """QKV 注意力 (legacy) — split head 在 chunk 之前。"""
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight, dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionPool2d(nn.Module):
    """CLIP 风格的 2D 注意力池化。"""
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


# ---------------------------------------------------------------------------
# UNet 构建块
# ---------------------------------------------------------------------------

class TimestepBlock(nn.Module):
    """forward 接受 timestep embedding 作为第二个参数的模块。"""
    @abstractmethod
    def forward(self, x: th.Tensor, emb: th.Tensor):
        ...


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """按序传递 timestep embedding 给子模块。"""
    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """条件残差块 — 带 timestep embedding 调制。

    支持 scale-shift norm (FiLM) 和 additive embedding 两种调制方式。
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        use_group_norm: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_group_norm = use_group_norm

        self.in_layers = nn.Sequential(
            normalization(channels) if self.use_group_norm else nn.Identity(),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels) if self.use_group_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """空间自注意力块 — 兼容 N-d (1D/2D/3D)。"""
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
        use_group_norm: bool = True,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels) if use_group_norm else nn.Identity()
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_flat))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# OFTSR Guided UNet
# ---------------------------------------------------------------------------

class OFTSR_UNet(nn.Module):
    """条件 Flow Matching UNet — OFTSR 核心模型。

    基于 OpenAI Guided Diffusion UNet，用作条件 Flow Matching 的
    教师/学生模型。支持：
    - 低分辨率图像条件 (LR 作为额外输入通道)
    - Timestep embedding 调制
    - 可选类别条件
    - 梯度检查点

    Args:
        image_size: 输入图像尺寸 (H, W 相同)
        in_channels: 输入通道数
        model_channels: 基础通道数
        out_channels: 输出通道数
        num_res_blocks: 每层残差块数
        attention_resolutions: 使用注意力的分辨率 (下采样级别)
        channel_mult: 各层通道倍率
        use_cond: 是否使用条件输入 (LR concat)
        use_input_skip: 是否使用输入跳跃连接 (x + out)
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: set,
        dropout: float = 0.0,
        channel_mult: tuple = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        use_group_norm: bool = True,
        use_attention: bool = True,
        use_input_skip: bool = True,
        use_cond: bool = False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_group_norm = use_group_norm
        self.use_attention = use_attention
        self.use_input_skip = use_input_skip

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = self.ch = input_ch = int(channel_mult[0] * model_channels)
        self.dims = dims
        self.use_cond = use_cond

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_group_norm=self.use_group_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions and self.use_attention:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            use_group_norm=self.use_group_norm,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout,
                            out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            use_group_norm=self.use_group_norm,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_group_norm=self.use_group_norm,
            ),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint,
                num_heads=num_heads, num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                use_group_norm=self.use_group_norm,
            ) if self.use_attention else nn.Identity(),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_group_norm=self.use_group_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers: list[nn.Module] = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_group_norm=self.use_group_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions and self.use_attention:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            use_group_norm=self.use_group_norm,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch, time_embed_dim, dropout,
                            out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            use_group_norm=self.use_group_norm,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch) if self.use_group_norm else nn.Identity(),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        y: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """前向传播。

        Args:
            x: [N x C x H x W] 输入 (噪声增强后的 LR 图像)
            timesteps: [N] 时间步
            y: [N] 类别标签 (可选)
        Returns:
            [N x C x H x W] 预测的速度场 (velocity field)
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        return self.out(h) + x if self.use_input_skip else self.out(h)


class OFTSR_SuperResModel(OFTSR_UNet):
    """OFTSR 超分辨率模型 — 将 LR 图像作为条件输入。

    将 LR 图像上采样后与输入 concat，输入通道数翻倍。
    """

    def __init__(self, image_size: int, in_channels: int, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        low_res: Optional[th.Tensor] = None,
        **kwargs,
    ) -> th.Tensor:
        _, _, new_height, new_width = x.shape
        if low_res is None:
            raise ValueError("low_res is required for OFTSR_SuperResModel.forward")
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)
