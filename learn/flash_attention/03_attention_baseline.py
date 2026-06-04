"""Step 03: materialized scaled dot-product attention in Triton."""

from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

CHECK_ONLY = False

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


def require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("This lesson requires Triton.")
    if not torch.cuda.is_available():
        raise RuntimeError("This lesson requires a CUDA device.")


def next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()



@triton.jit
def _scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    sm_scale: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    qk_base = pid_bh * N_CTX * HEAD_DIM
    q = tl.load(
        q_ptr + qk_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    k = tl.load(
        k_ptr + qk_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    scores = tl.dot(q, tl.trans(k)) * sm_scale
    valid = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
    if CAUSAL:
        valid = valid & (offs_m[:, None] >= offs_n[None, :])
    scores = tl.where(valid, scores, -float("inf"))

    scores_base = pid_bh * N_CTX * N_CTX
    tl.store(
        scores_ptr + scores_base + offs_m[:, None] * N_CTX + offs_n[None, :],
        scores,
        mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX),
    )

@triton.jit
def _row_softmax_kernel(
    scores_ptr,
    probs_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row = tl.load(
        scores_ptr + row_id * n_cols + offsets,
        mask=mask,
        other=-float("inf"),
    )
    row = row - tl.max(row, axis=0)
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    tl.store(probs_ptr + row_id * n_cols + offsets, output, mask=mask)

@triton.jit
def _pv_kernel(
    probs_ptr,
    v_ptr,
    out_ptr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    probs_base = pid_bh * N_CTX * N_CTX
    v_base = pid_bh * N_CTX * HEAD_DIM
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        cols = start_n + offs_n
        probs = tl.load(
            probs_ptr + probs_base + offs_m[:, None] * N_CTX + cols[None, :],
            mask=(offs_m[:, None] < N_CTX) & (cols[None, :] < N_CTX),
            other=0.0,
        )
        v = tl.load(
            v_ptr + v_base + cols[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(cols[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        acc += tl.dot(probs.to(tl.float32), v.to(tl.float32))

    tl.store(
        out_ptr + v_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        acc,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


def triton_attention_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    return_attention: bool = False,
    block_m: int = 32,
    block_n: int = 32,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Materialize S and P, but compute all three phases with Triton kernels."""
    require_triton_cuda()
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must have shape [batch, heads, seq_len, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape in this teaching example")

    batch, heads, seq_len, head_dim = q.shape
    block_d = next_power_of_2(head_dim)
    if block_d > 128:
        raise ValueError("this teaching kernel supports head_dim <= 128")

    q_contig = q.contiguous()
    k_contig = k.contiguous()
    v_contig = v.contiguous()
    bh = batch * heads

    scores = torch.empty((bh, seq_len, seq_len), device=q.device, dtype=torch.float32)
    probs = torch.empty_like(scores)
    out = torch.empty_like(q_contig)

    _scores_kernel[
        (triton.cdiv(seq_len, block_m), triton.cdiv(seq_len, block_n), bh)
    ](
        q_contig,
        k_contig,
        scores,
        sm_scale=1.0 / math.sqrt(head_dim),
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        CAUSAL=causal,
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )

    softmax_block = next_power_of_2(seq_len)
    _row_softmax_kernel[(bh * seq_len,)](
        scores,
        probs,
        seq_len,
        BLOCK_SIZE=softmax_block,
        num_warps=4 if softmax_block >= 1024 else 1,  # pyright: ignore[reportCallIssue]
    )

    _pv_kernel[(triton.cdiv(seq_len, block_m), bh)](
        probs,
        v_contig,
        out,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )

    if return_attention:
        return out, probs.reshape(batch, heads, seq_len, seq_len)
    return out, None


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Reference only: correctness checking, not the implementation being taught."""
    batch, heads, seq_len, head_dim = q.shape
    del batch, heads

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
    scores = scores * (1.0 / math.sqrt(head_dim))
    if causal:
        mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def estimate_attention_intermediate_bytes(
    batch: int,
    heads: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Estimate bytes for materialized scores and probabilities."""
    element_size = torch.empty((), dtype=dtype).element_size()
    one_matrix = batch * heads * seq_len * seq_len * element_size
    return 2 * one_matrix


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def check_correctness(causal: bool = True) -> float:
    require_triton_cuda()
    torch.manual_seed(3)
    q = torch.randn(2, 3, 65, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 3, 65, 32, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 3, 65, 32, device="cuda", dtype=torch.float16)
    expected = reference_attention(q, k, v, causal=causal)
    actual, _ = triton_attention_baseline(
        q,
        k,
        v,
        causal=causal,
        block_m=16,
        block_n=32,
    )
    return (actual.float() - expected.float()).abs().max().item()


def demo() -> None:
    require_triton_cuda()
    torch.manual_seed(4)
    q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16)
    out, probs = triton_attention_baseline(q, k, v, causal=True, return_attention=True)

    print(f"out shape: {tuple(out.shape)}")
    print(f"probs shape: {tuple(probs.shape) if probs is not None else None}")
    print(f"non-causal max error: {check_correctness(causal=False):.3e}")
    print(f"causal max error: {check_correctness(causal=True):.3e}")

    for seq_len in [1024, 2048, 4096, 8192]:
        num_bytes = estimate_attention_intermediate_bytes(
            batch=1,
            heads=32,
            seq_len=seq_len,
            dtype=torch.float16,
        )
        print(f"S + P memory at N={seq_len}: {format_bytes(num_bytes)}")


def main() -> None:
    error = max(check_correctness(causal=False), check_correctness(causal=True))
    if CHECK_ONLY:
        print(f"max error: {error:.3e}")
        return

    demo()


if __name__ == "__main__":
    main()
