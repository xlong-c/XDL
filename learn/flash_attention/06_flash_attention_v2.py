"""Step 06: a teaching FlashAttention-2 style forward kernel in Triton."""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import triton
    import triton.language as tl
else:
    try:
        import triton
        import triton.language as tl
    except Exception:
        triton = None
        tl = None  # type: ignore[assignment]


def next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("This lesson requires Triton.")
    if not torch.cuda.is_available():
        raise RuntimeError("This lesson requires a CUDA device.")


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Materialized reference attention."""
    batch, heads, seq_len, head_dim = q.shape
    del batch, heads

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
    scores = scores * (1.0 / math.sqrt(head_dim))
    if causal:
        mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)



@triton.jit
def _flash_attention_v2_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    sm_scale: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    base = pid_bh * N_CTX * HEAD_DIM

    q = tl.load(
        q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        kv_offsets = start_n + offs_n
        k = tl.load(
            k_ptr + base + kv_offsets[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(kv_offsets[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        v = tl.load(
            v_ptr + base + kv_offsets[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(kv_offsets[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * sm_scale
        valid = (offs_m[:, None] < N_CTX) & (kv_offsets[None, :] < N_CTX)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= kv_offsets[None, :])
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        has_values = block_max > -float("inf")
        safe_block_max = tl.where(has_values, block_max, 0.0)
        probs = tl.exp(scores - safe_block_max[:, None])
        probs = tl.where(valid, probs, 0.0)
        block_sum = tl.sum(probs, axis=1)

        new_row_max = tl.maximum(row_max, block_max)
        safe_new_row_max = tl.where(new_row_max > -float("inf"), new_row_max, 0.0)
        old_scale = tl.exp(row_max - safe_new_row_max)
        old_scale = tl.where(row_max > -float("inf"), old_scale, 0.0)
        block_scale = tl.exp(safe_block_max - safe_new_row_max)
        block_scale = tl.where(has_values, block_scale, 0.0)

        acc = acc * old_scale[:, None]
        acc = acc + tl.dot(probs.to(tl.float32), v.to(tl.float32)) * block_scale[
            :, None
        ]
        row_sum = row_sum * old_scale + block_sum * block_scale
        row_max = new_row_max

    denom = tl.where(row_sum == 0.0, 1.0, row_sum)
    out = acc / denom[:, None]
    tl.store(
        out_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


def triton_flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Forward-only FlashAttention-2 style attention."""
    require_triton_cuda()
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must have shape [batch, heads, seq_len, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape in this teaching example")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block sizes must be positive")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, v must be CUDA tensors")

    batch, heads, seq_len, head_dim = q.shape
    block_d = next_power_of_2(head_dim)
    if block_d > 128:
        raise ValueError("this teaching kernel supports head_dim <= 128")

    q_contig = q.contiguous()
    k_contig = k.contiguous()
    v_contig = v.contiguous()
    out = torch.empty_like(q_contig)

    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    _flash_attention_v2_fwd_kernel[grid](
        q_contig,
        k_contig,
        v_contig,
        out,
        sm_scale=1.0 / math.sqrt(head_dim),
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        CAUSAL=causal,
        num_warps=4 if block_d <= 64 else 8,  # pyright: ignore[reportCallIssue]
    )
    return out


def check_correctness(causal: bool = True) -> float:
    require_triton_cuda()
    device = "cuda"
    dtype = torch.float16

    torch.manual_seed(7)
    q = torch.randn(2, 2, 129, 64, device=device, dtype=dtype)
    k = torch.randn(2, 2, 129, 64, device=device, dtype=dtype)
    v = torch.randn(2, 2, 129, 64, device=device, dtype=dtype)
    actual = triton_flash_attention_v2(q, k, v, causal=causal, block_m=32, block_n=64)
    expected = reference_attention(q, k, v, causal=causal)
    return (actual.float() - expected.float()).abs().max().item()


def demo() -> None:
    require_triton_cuda()
    print("backend: triton")
    print(f"non-causal max error: {check_correctness(causal=False):.3e}")
    print(f"causal max error: {check_correctness(causal=True):.3e}")
    print("This is a forward-only teaching kernel, not a production FA2 package.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    error = max(check_correctness(causal=False), check_correctness(causal=True))
    if args.check_only:
        print(f"max error: {error:.3e}")
        return

    demo()


if __name__ == "__main__":
    main()
