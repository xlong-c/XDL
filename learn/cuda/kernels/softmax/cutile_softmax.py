# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch

# example-begin imports
import cuda.tile as ct
# example-end imports


# example-begin softmax
@ct.kernel
def softmax(input, output, B: ct.Constant[int], N: ct.Constant[int]):
    rows = ct.load(input, index=(ct.bid(0), 0), shape=(B, N))
    numerator = ct.exp(rows - ct.max(rows, axis=1, keepdims=True))
    denominator = ct.sum(numerator, axis=1, keepdims=True)
    ct.store(output, index=(ct.bid(0), 0), tile=numerator / denominator)
# example-end softmax


@ct.kernel
def softmax_per_row(input, output,
                    num_rows: ct.Constant[int],
                    num_cols: ct.Constant[int]):
    bidx = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    for i in range(bidx, num_rows, num_blocks):
        row = ct.load(input, index=(i, 0), shape=(1, num_cols))
        numerator = ct.exp(row - ct.max(row, axis=1, keepdims=True))
        denominator = ct.sum(numerator, axis=1, keepdims=True)
        ct.store(output, index=(i, 0), tile=numerator / denominator)


def softmax_gold(input):
    max_input = torch.max(input, dim=1, keepdim=True).values
    exps = torch.exp(input - max_input)
    sum_exps = torch.sum(exps, dim=1, keepdim=True)
    return exps / sum_exps


def benchmark_softmax(B=1024, N=1024, num_iters=100, warmup_iters=10):
    """Benchmark cutile softmax vs torch softmax"""
    import time

    # Create test data
    input_tensor = torch.randn(B, N, device='cuda')
    output_cutile = torch.empty_like(input_tensor)
    output_torch = torch.empty_like(input_tensor)

    print(f"Benchmarking softmax: B={B}, N={N}, iters={num_iters}")
    print(f"Input shape: {input_tensor.shape}")
    print("-" * 60)

    # Determine grid size (use min(B, 1024) blocks)
    grid_size = min(B, 1024)

    # Use torch CUDA stream
    stream = torch.cuda.Stream()

    # Get GPU compute capability
    compute_capability = torch.cuda.get_device_capability()
    arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
    print(f"GPU architecture: {arch}")

    # cutile only supports sm_100+ (datacenter GPUs like H100/Blackwell)
    # Check if current GPU is supported
    supported_archs = [(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)]  # sm_100, sm_103, sm_110, sm_120, sm_121
    is_supported = tuple(compute_capability) in supported_archs

    if is_supported:
        print(f"GPU {arch} is supported by cutile")
    else:
        print(f"NOTE: GPU {arch} is NOT supported by cutile (requires sm_100+)")
        print("      Running torch softmax benchmark only...")

    # Benchmark torch softmax
    start_time = time.time()
    for _ in range(num_iters):
        output_torch = softmax_gold(input_tensor)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / num_iters * 1000  # ms

    if is_supported:
        # Warmup cutile
        for _ in range(warmup_iters):
            ct.launch(stream, (grid_size,), softmax_per_row,
                      (input_tensor, output_cutile, B, N))
        torch.cuda.synchronize()

        # Benchmark cutile softmax
        start_time = time.time()
        for _ in range(num_iters):
            ct.launch(stream, (grid_size,), softmax_per_row,
                      (input_tensor, output_cutile, B, N))
        torch.cuda.synchronize()
        cutile_time = (time.time() - start_time) / num_iters * 1000  # ms

        # Verify correctness
        error = torch.max(torch.abs(output_cutile - output_torch)).item()
        print(f"Max error: {error:.2e}")

        # Print results
        print(f"Cutile softmax time: {cutile_time:.4f} ms")
        print(f"Torch softmax time:  {torch_time:.4f} ms")
        print(f"Speedup: {torch_time / cutile_time:.2f}x")
        print("-" * 60)

        return cutile_time, torch_time
    else:
        print(f"Torch softmax time:  {torch_time:.4f} ms")
        print("-" * 60)
        print("To run cutile benchmark, use a datacenter GPU (H100/H200/Blackwell)")

        return None, torch_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark cutile vs torch softmax")
    parser.add_argument("--B", type=int, default=1024, help="Number of rows")
    parser.add_argument("--N", type=int, default=1024, help="Number of columns")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")

    args = parser.parse_args()

    print("=" * 60)
    print("Cutile Softmax vs Torch Softmax Benchmark")
    print("=" * 60)

    benchmark_softmax(
        B=args.B,
        N=args.N,
        num_iters=args.iters,
        warmup_iters=args.warmup
    )

