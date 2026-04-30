"""Step 02: online softmax implemented as a Triton streaming kernel."""

from __future__ import annotations

import argparse
from typing import Tuple, TYPE_CHECKING

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



@triton.jit
def _online_softmax_kernel(
    x_ptr,
    out_ptr,
    row_max_ptr,
    row_sum_ptr,
    n_cols: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.full((), 0.0, tl.float32)

    for start in range(0, n_cols, BLOCK_N):
        cols = start + offsets
        mask = cols < n_cols
        block = tl.load(
            x_ptr + row_id * n_cols + cols,
            mask=mask,
            other=-float("inf"),
        )

        block_max = tl.max(block, axis=0)
        block_exp = tl.exp(block - block_max)
        block_exp = tl.where(mask, block_exp, 0.0)
        block_sum = tl.sum(block_exp, axis=0)

        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        old_scale = tl.where(running_max > -float("inf"), old_scale, 0.0)
        block_scale = tl.exp(block_max - new_max)
        running_sum = running_sum * old_scale + block_sum * block_scale
        running_max = new_max

    for start in range(0, n_cols, BLOCK_N):
        cols = start + offsets
        mask = cols < n_cols
        block = tl.load(
            x_ptr + row_id * n_cols + cols,
            mask=mask,
            other=-float("inf"),
        )
        probs = tl.exp(block - running_max) / running_sum
        tl.store(out_ptr + row_id * n_cols + cols, probs, mask=mask)

    tl.store(row_max_ptr + row_id, running_max)
    tl.store(row_sum_ptr + row_id, running_sum)


def triton_online_softmax(
    x: torch.Tensor,
    block_n: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact softmax by streaming each row through Triton blocks."""
    require_triton_cuda()
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    if block_n <= 0 or block_n & (block_n - 1) != 0:
        raise ValueError("block_n must be a positive power of two")

    x_2d = x.contiguous().reshape(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape
    out = torch.empty_like(x_2d)
    row_max = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    row_sum = torch.empty(n_rows, device=x.device, dtype=torch.float32)

    _online_softmax_kernel[(n_rows,)](
        x_2d,
        out,
        row_max,
        row_sum,
        n_cols,
        BLOCK_N=block_n,
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )
    return out.reshape_as(x), row_max.reshape(*x.shape[:-1], 1), row_sum.reshape(
        *x.shape[:-1],
        1,
    )


def check_correctness() -> float:
    require_triton_cuda()
    torch.manual_seed(1)
    x = torch.randn(7, 997, device="cuda") * 9.0
    actual, _, _ = triton_online_softmax(x, block_n=64)
    expected = torch.softmax(x, dim=-1)
    return (actual - expected).abs().max().item()


def demo() -> None:
    require_triton_cuda()
    torch.manual_seed(2)
    x = torch.randn(2, 20, device="cuda") * 4.0
    actual, row_max, row_sum = triton_online_softmax(x, block_n=8)
    expected = torch.softmax(x, dim=-1)

    print(f"online m:\n{row_max.cpu()}")
    print(f"online l:\n{row_sum.cpu()}")
    print(f"sum(probs): {actual.sum(dim=-1).cpu()}")
    print(f"max error vs torch.softmax reference: {(actual - expected).abs().max().item():.3e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    error = check_correctness()
    if args.check_only:
        print(f"max error: {error:.3e}")
        return

    demo()


if __name__ == "__main__":
    main()
