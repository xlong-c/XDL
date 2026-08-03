from typing import Tuple

import torch
from torch import nn


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(x.dtype)


def sinkhorn_normalize(logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """
    对齐 SGLang 的数值路径:
    1. 逐行减 max 做稳定化
    2. exp 转成正矩阵
    3. 先做一轮行归一化 + 列归一化
    4. 再做若干轮行列交替归一化
    """
    if logits.dim() < 2 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError(f"expected [..., N, N] logits, got {tuple(logits.shape)}")

    matrix = logits - logits.max(dim=-1, keepdim=True).values
    matrix = torch.exp(matrix)
    matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(eps) + eps
    matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(eps)

    for _ in range(max(iters - 1, 0)):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(eps)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(eps)

    return matrix


def split_mixes_with_sinkhorn(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mixes.dim() != 3:
        raise ValueError(f"expected mixes shape [B, L, M], got {tuple(mixes.shape)}")

    pre_logits = mixes[:, :, :hc_mult] * scale[0].float() + base[:hc_mult].float()
    post_logits = (
        mixes[:, :, hc_mult : 2 * hc_mult] * scale[1].float()
        + base[hc_mult : 2 * hc_mult].float()
    )
    comb_logits = (
        mixes[:, :, 2 * hc_mult :] * scale[2].float() + base[2 * hc_mult :].float()
    )

    pre_mix = torch.sigmoid(pre_logits) + eps
    post_mix = 2.0 * torch.sigmoid(post_logits)
    comb_mix = sinkhorn_normalize(
        comb_logits.view(*mixes.shape[:2], hc_mult, hc_mult),
        iters=sinkhorn_iters,
        eps=eps,
    )
    return pre_mix, post_mix, comb_mix


class ManifoldHyperConnection(nn.Module):
    """
    学习用的单一 canonical mHC 实现。

    residual streams:
        [B, L, N, D]
        B: batch_size
        L: seq_len
        N: hyper-connection expansion rate
        D: hidden dim
    """

    def __init__(
        self,
        dim: int,
        rate: int,
        layer_id: int,
        sinkhorn_iters: int,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rate = rate
        self.layer_id = layer_id
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

        self.flat_dim = dim * rate
        self.mix_dim = rate * rate + 2 * rate

        self.prenorm = UnweightedRMSNorm(eps=eps)
        self.mix_weight = nn.Parameter(torch.zeros(self.flat_dim, self.mix_dim))
        self.mix_scale = nn.Parameter(torch.ones(3) * 0.01)
        self.mix_base = nn.Parameter(torch.zeros(self.mix_dim))

    def _project_mixes(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.dim() != 4:
            raise ValueError(f"expected residual shape [B, L, N, D], got {tuple(residual.shape)}")
        if residual.shape[-2] != self.rate or residual.shape[-1] != self.dim:
            raise ValueError(
                f"expected residual shape [B, L, {self.rate}, {self.dim}], got {tuple(residual.shape)}"
            )

        flat = residual.reshape(*residual.shape[:2], self.flat_dim).float()
        flat = self.prenorm(flat)
        return flat @ self.mix_weight.float()

    def pre(self, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        输入 residual streams，输出:
        - post_mix: [B, L, N, 1]
        - comb_mix: [B, L, N, N]
        - layer_input: [B, L, D]
        """
        mixes = self._project_mixes(residual)
        pre_mix, post_mix, comb_mix = split_mixes_with_sinkhorn(
            mixes=mixes,
            scale=self.mix_scale,
            base=self.mix_base,
            hc_mult=self.rate,
            sinkhorn_iters=self.sinkhorn_iters,
            eps=self.eps,
        )
        layer_input = (pre_mix.to(residual.dtype).unsqueeze(-1) * residual).sum(dim=-2)
        return post_mix.unsqueeze(-1), comb_mix, layer_input

    def post(
        self,
        layer_output: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
    ) -> torch.Tensor:
        """
        对齐论文 / HF / SGLang:
            out = comb^T @ residual + post * layer_output
        """
        if layer_output.shape[:2] != residual.shape[:2] or layer_output.shape[-1] != residual.shape[-1]:
            raise ValueError(
                f"expected layer_output shape [B, L, {self.dim}] matching residual, got {tuple(layer_output.shape)}"
            )

        residual_mix = torch.matmul(comb_mix.to(residual.dtype).transpose(-1, -2), residual)
        layer_mix = post_mix.to(layer_output.dtype) * layer_output.unsqueeze(-2)
        return residual_mix + layer_mix


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class DecoderBlockMHC(nn.Module):
    """
    最小可学习版 Decoder block。

    结构故意保持朴素:
    1. attn_hc.pre -> attention_like -> attn_hc.post
    2. ffn_hc.pre -> ffn -> ffn_hc.post

    这里的 attention_like 用线性层代替，目的是把 mHC 的数据流讲清楚。
    """

    def __init__(
        self,
        dim: int,
        rate: int,
        layer_id: int,
        sinkhorn_iters: int,
        ffn_hidden_dim: int | None = None,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if ffn_hidden_dim is None:
            ffn_hidden_dim = dim * 4

        self.attn_hc = ManifoldHyperConnection(
            dim=dim,
            rate=rate,
            layer_id=layer_id,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
        )
        self.attention_like = nn.Linear(dim, dim)
        self.ffn_hc = ManifoldHyperConnection(
            dim=dim,
            rate=rate,
            layer_id=layer_id,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
        )
        self.ffn = FeedForward(dim=dim, hidden_dim=ffn_hidden_dim)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        post_mix, comb_mix, layer_input = self.attn_hc.pre(residual)
        layer_output = self.attention_like(layer_input)
        residual = self.attn_hc.post(layer_output, residual, post_mix, comb_mix)

        post_mix, comb_mix, layer_input = self.ffn_hc.pre(residual)
        layer_output = self.ffn(layer_input)
        residual = self.ffn_hc.post(layer_output, residual, post_mix, comb_mix)
        return residual


def _demo() -> None:
    dim = 512
    rate = 8
    layer_id = 10
    batch_size = 1
    seq_len = 32
    sinkhorn_iters = 20

    residual = torch.randn(batch_size, seq_len, rate, dim)
    block = DecoderBlockMHC(
        dim=dim,
        rate=rate,
        layer_id=layer_id,
        sinkhorn_iters=sinkhorn_iters,
    )
    out = block(residual)
    print("out", out.shape)


if __name__ == "__main__":
    _demo()
