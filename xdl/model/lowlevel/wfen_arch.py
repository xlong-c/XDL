"""WFEN: Wavelet-based Feature Enhancement Network for Face Super-Resolution.

ACM MM 2024 — 小波 + Transformer 人脸超分 (128→1024, 8x).
https://github.com/PRIS-CV/WFEN
"""

import numbers

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


# ── LayerNorm helpers ──────────────────────────────────────────

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
        self.bias = nn.Parameter(torch.zeros(torch.Size(normalized_shape)))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        body = WithBias_LayerNorm if LayerNorm_type == "WithBias" else BiasFree_LayerNorm
        self.body = body(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ── FeedForward ────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1,
            padding=1, groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ── Attention modules ─────────────────────────────────────────

class GSA(nn.Module):
    """Global Self-Attention."""

    def __init__(self, channels, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, stride=1,
            padding=1, groups=channels * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = torch.relu(attn)
            out = attn @ v
            y = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
            y = rearrange(y, "b (head c) h w -> b (c head) h w", head=self.num_heads, h=h, w=w)
            return self.project_out(y), attn
        else:
            v = rearrange(x, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            out = prev_atns @ v
            y = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
            y = rearrange(y, "b (head c) h w -> b (c head) h w", head=self.num_heads, h=h, w=w)
            return self.project_out(y)


class RSA(nn.Module):
    """Regional Self-Attention with window partitioning."""

    def __init__(self, channels, num_heads, shifts=1, window_sizes=8, bias=False):
        super().__init__()
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, stride=1,
            padding=1, groups=channels * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        wsize = self.window_sizes
        if prev_atns is None:
            x_ = x
            if self.shifts > 0:
                x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
            qkv = self.qkv_dwconv(self.qkv(x_))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)
            k = rearrange(k, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)
            v = rearrange(v, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = torch.relu(attn)
            out = v @ attn
            out = rearrange(out, "b (h w) (dh dw) c -> b c (h dh) (w dw)",
                            h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
            if self.shifts > 0:
                out = torch.roll(out, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            return self.project_out(out), attn
        else:
            x_ = x
            if self.shifts > 0:
                x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
            v = rearrange(x_, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)
            out = v @ prev_atns
            out = rearrange(out, "b (h w) (dh dw) c -> b c (h dh) (w dw)",
                            h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
            if self.shifts > 0:
                out = torch.roll(out, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            return self.project_out(out)


class FDT(nn.Module):
    """Full-Domain Transformer (RSA + GSA)."""

    def __init__(self, inp_channels, window_sizes, shifts, num_heads,
                 shared_depth=1, ffn_expansion_factor=2.66):
        super().__init__()

        modules_ffd, modules_att, modules_norm = {}, {}, {}
        for i in range(shared_depth):
            modules_ffd[f"ffd{i}"] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modules_att[f"att_{i}"] = RSA(inp_channels, num_heads, shifts, window_sizes)
            modules_norm[f"norm_{i}"] = LayerNorm(inp_channels, "WithBias")
            modules_norm[f"norm_{i + 2}"] = LayerNorm(inp_channels, "WithBias")
        self.modules_ffd = nn.ModuleDict(modules_ffd)
        self.modules_att = nn.ModuleDict(modules_att)
        self.modules_norm = nn.ModuleDict(modules_norm)

        modulec_ffd, modulec_att, modulec_norm = {}, {}, {}
        for i in range(shared_depth):
            modulec_ffd[f"ffd{i}"] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modulec_att[f"att_{i}"] = GSA(inp_channels, num_heads)
            modulec_norm[f"norm_{i}"] = LayerNorm(inp_channels, "WithBias")
            modulec_norm[f"norm_{i + 2}"] = LayerNorm(inp_channels, "WithBias")
        self.modulec_ffd = nn.ModuleDict(modulec_ffd)
        self.modulec_att = nn.ModuleDict(modulec_att)
        self.modulec_norm = nn.ModuleDict(modulec_norm)

    def forward(self, x):
        atn = None
        for i in range(len(self.modules_ffd)):
            if i == 0:
                x_, atn = self.modules_att[f"att_{i}"](
                    self.modules_norm[f"norm_{i}"](x), None)
                x = self.modules_ffd[f"ffd{i}"](
                    self.modules_norm[f"norm_{i + 2}"](x_ + x)) + x_
            else:
                x_ = self.modules_att[f"att_{i}"](
                    self.modules_norm[f"norm_{i}"](x), atn)
                x = self.modules_ffd[f"ffd{i}"](
                    self.modules_norm[f"norm_{i + 2}"](x_ + x)) + x_

        for i in range(len(self.modulec_ffd)):
            if i == 0:
                x_, atn = self.modulec_att[f"att_{i}"](
                    self.modulec_norm[f"norm_{i}"](x), None)
                x = self.modulec_ffd[f"ffd{i}"](
                    self.modulec_norm[f"norm_{i + 2}"](x_ + x)) + x_
            else:
                x = self.modulec_att[f"att_{i}"](
                    self.modulec_norm[f"norm_{i}"](x), atn)
                x = self.modulec_ffd[f"ffd{i}"](
                    self.modulec_norm[f"norm_{i + 2}"](x_ + x)) + x_

        return x


# ── Haar Wavelet ───────────────────────────────────────────────

class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super().__init__()
        self.in_channels = in_channels
        self.haar_weights = torch.ones(4, 1, 2, 2)
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1
        self.haar_weights = torch.cat([self.haar_weights] * in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape(x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2)
            out = out.transpose(1, 2).reshape(x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2)
            return out
        else:
            out = x.reshape(x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3])
            out = out.transpose(1, 2).reshape(x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)


# ── Wavelet Feature Up/Down ────────────────────────────────────

class WFU(nn.Module):
    """Wavelet Feature Upgrade (decoder)."""

    def __init__(self, dim_big, dim_small):
        super().__init__()
        self.dim = dim_big
        self.HaarWavelet = HaarWavelet(dim_big, grad=False)
        self.InverseHaarWavelet = HaarWavelet(dim_big, grad=False)
        self.RB = nn.Sequential(
            nn.Conv2d(dim_big, dim_big, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim_big, dim_big, 3, 1, 1),
        )
        self.channel_tranformation = nn.Sequential(
            nn.Conv2d(dim_big + dim_small, dim_big + dim_small // 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim_big + dim_small // 1, dim_big * 3, 1),
        )

    def forward(self, x_big, x_small):
        haar = self.HaarWavelet(x_big, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)
        hvd = self.RB(h + v + d)
        a_ = self.channel_tranformation(torch.cat([x_small, a], dim=1))
        return self.InverseHaarWavelet(torch.cat([hvd, a_], dim=1), rev=True)


class WFD(nn.Module):
    """Wavelet Feature Downsample (encoder)."""

    def __init__(self, dim_in, dim, need=False):
        super().__init__()
        self.need = need
        if need:
            self.first_conv = nn.Conv2d(dim_in, dim, kernel_size=1)
        self.HaarWavelet = HaarWavelet(dim if need else dim_in, grad=False)
        self.dim = dim if need else dim_in

    def forward(self, x):
        if self.need:
            x = self.first_conv(x)
        haar = self.HaarWavelet(x, rev=False)
        a = haar.narrow(1, 0, self.dim)
        hvd = haar.narrow(1, self.dim, self.dim) + haar.narrow(1, self.dim * 2, self.dim) + haar.narrow(1, self.dim * 3, self.dim)
        return a, hvd


# ── WFEN Main Model ───────────────────────────────────────────

class WFEN(nn.Module):
    """Wavelet-based Feature Enhancement Network.

    输入/输出同分辨率，用于人脸图像细节增强。
    SR 用法: 先 bicubic 上采样到目标尺寸，再送入 WFEN 增强细节。
    """

    def __init__(self, inchannel=3, min_ch=40, res_depth=6):
        super().__init__()
        self.min_ch = min_ch

        self.first_conv = nn.Conv2d(inchannel, min_ch, 3, 1, 1)

        self.HaarDownsample1 = WFD(min_ch, min_ch * 2, True)
        self.HaarDownsample2 = WFD(min_ch * 2, min_ch * 4, True)
        self.HaarDownsample3 = WFD(min_ch * 4, min_ch * 4, True)

        self.TransformerDown1 = nn.Sequential(
            FDT(min_ch, window_sizes=8, shifts=0, num_heads=4),
            FDT(min_ch, window_sizes=8, shifts=0, num_heads=4),
        )
        self.TransformerDown2 = nn.Sequential(
            FDT(min_ch * 2, window_sizes=4, shifts=1, num_heads=4),
        )
        self.TransformerDown3 = nn.Sequential(
            FDT(min_ch * 4, window_sizes=2, shifts=0, num_heads=8),
        )

        self.RB1 = nn.Sequential(
            nn.Conv2d(min_ch * 2, min_ch * 2, 1), nn.ReLU(),
            nn.Conv2d(min_ch * 2, min_ch * 2, 1),
        )
        self.RB2 = nn.Sequential(
            nn.Conv2d(min_ch * 4, min_ch * 4, 1), nn.ReLU(),
            nn.Conv2d(min_ch * 4, min_ch * 4, 1),
        )
        self.RB3 = nn.Sequential(
            nn.Conv2d(min_ch * 4, min_ch * 4, 1), nn.ReLU(),
            nn.Conv2d(min_ch * 4, min_ch * 4, 1),
        )

        self.Transformer0 = FDT(min_ch * 4, window_sizes=1, shifts=0, num_heads=8)
        Transformer = []
        for i in range(res_depth - 1):
            if i % 2 == 0:
                Transformer.append(FDT(min_ch * 4, window_sizes=1, shifts=1, num_heads=8))
            else:
                Transformer.append(FDT(min_ch * 4, window_sizes=1, shifts=0, num_heads=8))
        self.Transformer = nn.Sequential(*Transformer)

        self.TransformerUp1 = nn.Sequential(
            FDT(min_ch * 4, window_sizes=8, shifts=1, num_heads=8),
        )
        self.TransformerUp2 = nn.Sequential(
            FDT(min_ch * 2, window_sizes=4, shifts=0, num_heads=4),
        )
        self.TransformerUp3 = nn.Sequential(
            FDT(min_ch, window_sizes=2, shifts=1, num_heads=4),
            FDT(min_ch, window_sizes=2, shifts=0, num_heads=4),
        )

        self.HaarFeatureFusion1 = WFU(min_ch * 4, min_ch * 4)
        self.HaarFeatureFusion2 = WFU(min_ch * 2, min_ch * 4)
        self.HaarFeatureFusion3 = WFU(min_ch, min_ch * 2)

        self.out_conv = nn.Conv2d(min_ch, inchannel, 3, 1, 1)

    def forward(self, input_img):
        x_first = self.first_conv(input_img)

        # ── encoder ──
        x1 = self.TransformerDown1(x_first)
        x1_a, x1_hvd = self.HaarDownsample1(x1)

        x2 = self.TransformerDown2(x1_a)
        x2 = x2 + self.RB1(x1_hvd)
        x2_a, x2_hvd = self.HaarDownsample2(x2)

        x3 = self.TransformerDown3(x2_a)
        x3 = x3 + self.RB2(x2_hvd)
        x3_a, x3_hvd = self.HaarDownsample3(x3)

        # ── bottleneck ──
        x_trans0 = self.Transformer0(x3_a)
        x_trans = self.Transformer(x_trans0)
        x_trans = x_trans + self.RB3(x3_hvd)

        # ── decoder ──
        x_up1 = self.HaarFeatureFusion1(x3, x_trans)
        x_1 = self.TransformerUp1(x_up1)

        x_up2 = self.HaarFeatureFusion2(x2, x_1)
        x_2 = self.TransformerUp2(x_up2)

        x_up3 = self.HaarFeatureFusion3(x1, x_2)
        x_3 = self.TransformerUp3(x_up3)

        return self.out_conv(x_3 + x_first)
