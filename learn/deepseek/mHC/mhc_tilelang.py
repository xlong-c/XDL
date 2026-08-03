import functools
import math
from typing import Tuple

import tilelang
import tilelang.language as T
import torch
from torch import nn

from mhc import UnweightedRMSNorm

tilelang.set_log_level("WARNING")

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


@functools.cache
@tilelang.jit(pass_configs=PASS_CONFIGS)
def hc_split_sinkhorn_kernel(hc_mult: int, sinkhorn_iters: int, eps: float):
    num_tokens = T.symbolic("num_tokens")
    mix_dim = (2 + hc_mult) * hc_mult

    @T.prim_func
    def kernel(
        mixes: T.Tensor[(num_tokens, mix_dim), T.float32],
        hc_scale: T.Tensor[(3,), T.float32],
        hc_base: T.Tensor[(mix_dim,), T.float32],
        pre_mix: T.Tensor[(num_tokens, hc_mult), T.float32],
        post_mix: T.Tensor[(num_tokens, hc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, hc_mult, hc_mult), T.float32],
    ):
        with T.Kernel(num_tokens, threads=64) as i:
            mixes_shared = T.alloc_shared(mix_dim, T.float32)
            comb_frag = T.alloc_fragment((hc_mult, hc_mult), T.float32)

            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc_mult):
                pre_mix[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps

            for j in T.Parallel(hc_mult):
                post_mix[i, j] = 2 * T.sigmoid(
                    mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                )

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_frag[j, k] = (
                    mixes_shared[j * hc_mult + k + 2 * hc_mult] * hc_scale[2]
                    + hc_base[j * hc_mult + k + 2 * hc_mult]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)
            row_max = T.alloc_fragment(hc_mult, T.float32)

            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])

            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)

                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb_mix[i, :, :])

    return kernel


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mixes.device.type != "cuda":
        raise ValueError("TileLang kernels require CUDA inputs")
    if mixes.dtype != torch.float32:
        raise ValueError(f"expected float32 mixes, got {mixes.dtype}")

    pre_mix = mixes.new_empty(mixes.shape[0], hc_mult)
    post_mix = mixes.new_empty(mixes.shape[0], hc_mult)
    comb_mix = mixes.new_empty(mixes.shape[0], hc_mult, hc_mult)
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(
        mixes,
        hc_scale.float(),
        hc_base.float(),
        pre_mix,
        post_mix,
        comb_mix,
    )
    return pre_mix, post_mix, comb_mix


@functools.cache
@tilelang.jit(pass_configs=PASS_CONFIGS)
def hc_weighted_sum_kernel(hc_mult: int, hidden_size: int, h_blk: int = 1024):
    num_tokens = T.symbolic("num_tokens")
    block = math.gcd(hidden_size, h_blk)

    @T.prim_func
    def kernel(
        pre_mix: T.Tensor[(num_tokens, hc_mult), T.float32],
        residual: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ):
        with T.Kernel(num_tokens, threads=128) as i:
            pre_local = T.alloc_fragment(hc_mult, T.float32)
            T.copy(pre_mix[i, 0], pre_local)

            for i0_h in T.Pipelined(T.ceildiv(hidden_size, block), num_stages=2):
                xs = T.alloc_shared((hc_mult, block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, block), T.float32)
                ol = T.alloc_fragment(block, T.float32)
                T.clear(ol)

                T.copy(residual[i, 0, i0_h * block], xs)
                T.copy(xs, xl)

                for i_hc in T.serial(hc_mult):
                    pre = pre_local[i_hc]
                    for i1_h in T.Parallel(block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * block])

    return kernel


def hc_weighted_sum(pre_mix: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    if pre_mix.device.type != "cuda" or residual.device.type != "cuda":
        raise ValueError("TileLang kernels require CUDA inputs")

    num_tokens, hc_mult, hidden_size = residual.shape
    out = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device)
    kernel = hc_weighted_sum_kernel(hc_mult, hidden_size)
    kernel(pre_mix.float(), residual.bfloat16(), out)
    return out


@functools.cache
@tilelang.jit(pass_configs=PASS_CONFIGS)
def mhc_post_kernel(hc_mult: int, hidden_size: int, h_blk: int = 1024):
    num_tokens = T.symbolic("num_tokens")
    block = math.gcd(hidden_size, h_blk)

    @T.prim_func
    def kernel(
        comb_mix: T.Tensor[(num_tokens, hc_mult, hc_mult), T.float32],
        residual: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, hc_mult), T.float32],
        layer_output: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        out: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
    ):
        with T.Kernel(num_tokens, threads=128) as i:
            comb_local = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            post_local = T.alloc_fragment(hc_mult, T.float32)
            T.copy(comb_mix[i, 0, 0], comb_local)
            T.copy(post_mix[i, 0], post_local)

            for i0_h in T.Pipelined(T.ceildiv(hidden_size, block), num_stages=2):
                res_shared = T.alloc_shared((hc_mult, block), T.bfloat16)
                x_shared = T.alloc_shared((hc_mult, block), T.bfloat16)
                out_shared = T.alloc_shared(block, T.bfloat16)

                res_local = T.alloc_fragment((hc_mult, block), T.float32)
                out_local = T.alloc_fragment(block, T.float32)
                x_local = T.alloc_fragment((hc_mult, block), T.float32)

                T.copy(residual[i, 0, i0_h * block], res_shared)
                T.copy(res_shared, res_local)
                T.copy(layer_output[i, i0_h * block], out_shared)
                T.copy(out_shared, out_local)

                for i_hco, i1_h in T.Parallel(hc_mult, block):
                    x_local[i_hco, i1_h] = post_local[i_hco] * out_local[i1_h]
                    for i_hci in T.serial(hc_mult):
                        x_local[i_hco, i1_h] += comb_local[i_hci, i_hco] * res_local[i_hci, i1_h]

                T.copy(x_local, x_shared)
                T.copy(x_shared, out[i, 0, i0_h * block])

    return kernel


def mhc_post(
    layer_output: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    if layer_output.device.type != "cuda" or residual.device.type != "cuda":
        raise ValueError("TileLang kernels require CUDA inputs")

    num_tokens, hc_mult, hidden_size = residual.shape
    out = torch.empty_like(residual, dtype=torch.bfloat16)
    kernel = mhc_post_kernel(hc_mult, hidden_size)
    kernel(
        comb_mix.float(),
        residual.bfloat16(),
        post_mix.float(),
        layer_output.bfloat16(),
        out,
    )
    return out


class ManifoldHyperConnectionTileLang(nn.Module):
    """
    学习用的 SGLang 风格 TileLang 版 mHC。

    这个版本只保留最核心的三段优化:
    1. split + sinkhorn kernel
    2. weighted-sum kernel
    3. fused post kernel

    SGLang 还把 pre 再进一步 fuse 到一个大 kernel 里。
    这里先保留分段版本，避免教学版被 TileLang 的布局细节淹没。
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
        if residual.device.type != "cuda":
            raise ValueError("TileLang version expects CUDA residual inputs")
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
        outer_shape = residual.shape[:-2]
        mixes = self._project_mixes(residual).reshape(-1, self.mix_dim).contiguous()
        residual_flat = residual.reshape(-1, self.rate, self.dim).contiguous()

        pre_mix, post_mix, comb_mix = hc_split_sinkhorn(
            mixes=mixes,
            hc_scale=self.mix_scale,
            hc_base=self.mix_base,
            hc_mult=self.rate,
            sinkhorn_iters=self.sinkhorn_iters,
            eps=self.eps,
        )
        layer_input = hc_weighted_sum(pre_mix=pre_mix, residual=residual_flat)
        post_mix = post_mix.view(*outer_shape, self.rate, 1)
        comb_mix = comb_mix.view(*outer_shape, self.rate, self.rate)
        layer_input = layer_input.view(*outer_shape, self.dim).to(residual.dtype)
        return post_mix, comb_mix, layer_input

    def post(
        self,
        layer_output: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
    ) -> torch.Tensor:
        outer_shape = residual.shape[:-2]
        out = mhc_post(
            layer_output=layer_output.reshape(-1, self.dim).contiguous(),
            residual=residual.reshape(-1, self.rate, self.dim).contiguous(),
            post_mix=post_mix.reshape(-1, self.rate).contiguous(),
            comb_mix=comb_mix.reshape(-1, self.rate, self.rate).contiguous(),
        )
        return out.view(*outer_shape, self.rate, self.dim).to(residual.dtype)


def _demo() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skip TileLang demo")
        return

    dim = 128
    rate = 4
    layer_id = 0
    batch_size = 2
    seq_len = 8
    sinkhorn_iters = 8

    mhc = ManifoldHyperConnectionTileLang(
        dim=dim,
        rate=rate,
        layer_id=layer_id,
        sinkhorn_iters=sinkhorn_iters,
    ).cuda()
    sublayer = nn.Linear(dim, dim).cuda().bfloat16()

    residual = torch.randn(batch_size, seq_len, rate, dim, device="cuda", dtype=torch.bfloat16)
    post_mix, comb_mix, layer_input = mhc.pre(residual)
    layer_output = sublayer(layer_input)
    out = mhc.post(layer_output, residual, post_mix, comb_mix)
    print("out", out.shape, out.dtype, out.device)


if __name__ == "__main__":
    _demo()
