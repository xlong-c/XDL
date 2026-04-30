"""Step 05: row-wise fused softmax in Triton."""

from __future__ import annotations

import argparse
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


def num_warps_for_block(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 1024:
        return 4
    return 1



@triton.jit
def _softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row = tl.load(x_ptr + row_id * n_cols + offsets, mask=mask, other=-float("inf"))
    row = row - tl.max(row, axis=0)
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    tl.store(out_ptr + row_id * n_cols + offsets, output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension using Triton."""
    require_triton_cuda()
    if x.numel() == 0:
        return torch.empty_like(x)
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")

    x_2d = x.contiguous().reshape(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape
    block_size = next_power_of_2(n_cols)
    if block_size > 131072:
        raise ValueError("this teaching kernel only supports rows up to 131072 values")

    out = torch.empty_like(x_2d)
    _softmax_kernel[(n_rows,)](
        x_2d,
        out,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps_for_block(block_size),  # pyright: ignore[reportCallIssue]
    )
    return out.reshape_as(x)


def check_correctness() -> float:
    require_triton_cuda()
    device = "cuda"
    torch.manual_seed(6)
    x = torch.randn(17, 513, device=device) * 5.0
    actual = triton_softmax(x)
    expected = torch.softmax(x, dim=-1)
    return (actual - expected).abs().max().item()


def demo() -> None:
    require_triton_cuda()
    device = "cuda"
    x = torch.randn(4, 9, device=device)
    y = triton_softmax(x)
    print(f"device: {device}")
    print(f"row sums: {y.sum(dim=-1)}")
    print(f"max error vs torch.softmax: {check_correctness():.3e}")


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
