"""Step 01: naive softmax and safe softmax as Triton kernels."""

from __future__ import annotations

import argparse
from typing import Optional, Tuple, TYPE_CHECKING

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


def num_warps_for_block(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 1024:
        return 4
    return 1



@triton.jit
def _naive_softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row = tl.load(x_ptr + row_id * n_cols + offsets, mask=mask, other=-float("inf"))
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    tl.store(out_ptr + row_id * n_cols + offsets, output, mask=mask)

@triton.jit
def _safe_softmax_kernel(
    x_ptr,
    out_ptr,
    row_max_ptr,
    row_sum_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row = tl.load(x_ptr + row_id * n_cols + offsets, mask=mask, other=-float("inf"))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator

    tl.store(out_ptr + row_id * n_cols + offsets, output, mask=mask)
    tl.store(row_max_ptr + row_id, row_max)
    tl.store(row_sum_ptr + row_id, denominator)


def _launch_softmax(
    x: torch.Tensor,
    safe: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    require_triton_cuda()
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")

    x_2d = x.contiguous().reshape(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape
    block_size = next_power_of_2(n_cols)
    if block_size > 131072:
        raise ValueError("this teaching kernel only supports rows up to 131072 values")

    out = torch.empty_like(x_2d)
    row_max = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    row_sum = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    grid = (n_rows,)

    if safe:
        _safe_softmax_kernel[grid](
            x_2d,
            out,
            row_max,
            row_sum,
            n_cols,
            BLOCK_SIZE=block_size,
            num_warps=num_warps_for_block(block_size),  # pyright: ignore[reportCallIssue]
        )
        return out.reshape_as(x), row_max, row_sum

    _naive_softmax_kernel[grid](
        x_2d,
        out,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps_for_block(block_size),  # pyright: ignore[reportCallIssue]
    )
    return out.reshape_as(x), None, None


def triton_naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute softmax directly in Triton. This intentionally overflows."""
    out, _, _ = _launch_softmax(x, safe=False)
    return out


def triton_safe_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute numerically safe softmax in Triton."""
    out, _, _ = _launch_softmax(x, safe=True)
    return out


def triton_softmax_with_state(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Triton softmax output plus row max and shifted denominator."""
    out, row_max, row_sum = _launch_softmax(x, safe=True)
    if row_max is None or row_sum is None:
        raise RuntimeError("safe softmax state was not produced")
    return out, row_max.reshape(*x.shape[:-1], 1), row_sum.reshape(*x.shape[:-1], 1)


def check_correctness() -> float:
    require_triton_cuda()
    torch.manual_seed(0)
    x = torch.randn(8, 257, device="cuda") * 7.0
    expected = torch.softmax(x, dim=-1)
    actual = triton_safe_softmax(x)
    return (actual - expected).abs().max().item()


def demo_overflow() -> None:
    require_triton_cuda()
    x = torch.tensor(
        [[1000.0, 1001.0, 1002.0], [1.0, 2.0, 3.0]],
        device="cuda",
        dtype=torch.float32,
    )
    naive = triton_naive_softmax(x)
    probs, row_max, row_sum = triton_softmax_with_state(x)

    print("input:")
    print(x.cpu())
    print("\ntriton_naive_softmax(input):")
    print(naive.cpu())
    print("\ntriton_safe_softmax(input):")
    print(probs.cpu())
    print("\nrow max m:")
    print(row_max.cpu())
    print("\nshifted denominator l = sum(exp(x - m)):")
    print(row_sum.cpu())
    print(f"\nmax error vs torch.softmax reference: {check_correctness():.3e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    error = check_correctness()
    if args.check_only:
        print(f"max error: {error:.3e}")
        return

    torch.set_printoptions(precision=6, sci_mode=False)
    demo_overflow()


if __name__ == "__main__":
    main()
