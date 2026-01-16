"""
Softmax Performance Benchmark: CUDA vs PyTorch vs TileLang

Tests softmax implementations across different frameworks:
- CUDA: Custom optimized kernel with online reduction
- PyTorch: Native torch.nn.functional.softmax
- TileLang: DSL-based GPU kernel compiler
"""

import os
from typing import Optional, Callable, cast

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import tilelang as tl
import tilelang.language as T
from tilelang.profiler import do_bench

# Get the current directory and load CUDA extension
curr_dir = os.path.dirname(os.path.abspath(__file__))
cuda_file = os.path.join(curr_dir, "softmax.cu")
build_dir = os.path.join(curr_dir, "build")
os.makedirs(build_dir, exist_ok=True)

# Load CUDA extension
softmax_cuda = load(
    name="softmax_cuda",
    sources=[cuda_file],
    extra_cflags=["-DPYTORCH_EXTENSION"],
    extra_cuda_cflags=["-DPYTORCH_EXTENSION"],
    build_directory=build_dir,
    verbose=False,
)


# =============================================================================
# TileLang Softmax Kernel (with pipelining and exp2 optimization)
# =============================================================================

@tl.jit(out_idx=[1])  # type: ignore[return-value]
def tilelang_softmax_kernel(
    M: int,
    N: int,
    dtype: T.dtype = T.float16,  # type: ignore[assignment]
) -> Callable:
    """Optimized TileLang softmax with pipelined reduction and exp2."""
    BN = min(tl.next_power_of_2(N), 8192)
    NN = tl.cdiv(N, BN)
    accum_dtype = T.float32
    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        X: T.Tensor([M, N], dtype),  # type: ignore[valid-type]
        Y: T.Tensor([M, N], dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(M, threads=128) as (i_m):
            x = T.alloc_fragment([BN], dtype)
            y = T.alloc_fragment([BN], dtype)
            lse = T.alloc_fragment([1], accum_dtype)
            max_x = T.alloc_fragment([1], dtype)
            exp_x = T.alloc_fragment([BN], accum_dtype)
            sum_exp_x = T.alloc_fragment([1], accum_dtype)
            T.fill(lse, -T.infinity(accum_dtype))  # type: ignore[operator]

            # Forward pass: compute log-sum-exp
            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                T.reduce_max(x, max_x, dim=0, clear=True)

                for j in T.Parallel(BN):
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)

                T.reduce_sum(exp_x, sum_exp_x, dim=0, clear=True)
                lse[0] = max_x[0] * scale + T.log2(
                    T.exp2(lse[0] - max_x[0] * scale) + sum_exp_x[0]
                )

            # Backward pass: normalize
            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])
                T.copy(y, Y[i_m, i_n * BN : (i_n + 1) * BN])

    return main


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_torch(input_data: torch.Tensor) -> float:
    """Benchmark PyTorch native softmax using CUDA events."""
    result = do_bench(lambda: F.softmax(input_data, dim=-1), warmup=25, rep=100)
    if isinstance(result, list):
        return result[0] if result else 0.0
    return float(result) if result is not None else 0.0


def benchmark_cuda(input_data: torch.Tensor, output: torch.Tensor) -> float:
    """Benchmark custom CUDA softmax using CUDA events."""
    if softmax_cuda is None:
        raise RuntimeError("CUDA extension not loaded")
    result = do_bench(lambda: softmax_cuda.forward(input_data, output), warmup=25, rep=100)  # type: ignore[arg-type]
    if isinstance(result, list):
        return result[0] if result else 0.0
    return float(result) if result is not None else 0.0


def benchmark_tilelang(kernel: Callable, input_data: torch.Tensor) -> Optional[float]:
    """Benchmark TileLang softmax using its profiler."""
    try:
        result = do_bench(lambda: kernel(input_data), warmup=25, rep=100)
        if isinstance(result, list):
            return result[0] if result else None
        return float(result) if result is not None else None
    except Exception as e:
        print(f"  TileLang execution failed: {e}")
        return None


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(rows: int, cols: int, dtype: torch.dtype = torch.float16):
    """Run comprehensive benchmark for a single matrix size."""
    print(f"\n{'='*70}")
    print(f"Matrix: {rows} x {cols} | dtype: {dtype}")
    print(f"{'='*70}")

    input_data = torch.randn(rows, cols, device="cuda", dtype=dtype)
    output_cuda = torch.empty_like(input_data)

    # Compile TileLang kernel
    tilelang_kernel = None
    tilelang_time = None
    output_tilelang = None
    try:
        tl_dtype = T.float16 if dtype == torch.float16 else T.float32
        tilelang_kernel = tilelang_softmax_kernel(rows, cols, dtype=tl_dtype)  # type: ignore[arg-type]
        print("TileLang kernel: compiled successfully")
    except Exception as e:
        print(f"TileLang kernel: compilation failed - {str(e)[:80]}")

    # Warmup
    for _ in range(10):
        if dtype == torch.float32 and softmax_cuda is not None:
            softmax_cuda.forward(input_data, output_cuda)
        F.softmax(input_data, dim=-1)
        if tilelang_kernel is not None:
            try:
                output_tilelang = tilelang_kernel(input_data)
            except Exception as e:
                tilelang_kernel = None
                print(f"TileLang kernel: execution failed during warmup - {e}")
    torch.cuda.synchronize()

    # Benchmarks
    torch_time = benchmark_torch(input_data)

    # CUDA kernel only supports float32
    if dtype == torch.float32:
        cuda_time = benchmark_cuda(input_data, output_cuda)
    else:
        cuda_time = None

    if tilelang_kernel is not None:
        tilelang_time = benchmark_tilelang(tilelang_kernel, input_data)

    # Verification
    output_torch = F.softmax(input_data, dim=-1)

    print(f"\nResults:")
    print(f"  PyTorch:    {torch_time:8.4f} ms")
    if cuda_time is not None:
        cuda_diff = torch.abs(output_cuda - output_torch).max().item()
        print(f"  CUDA:       {cuda_time:8.4f} ms  (speedup: {torch_time/cuda_time:.2f}x)")
    else:
        print(f"  CUDA:       (skipped - float32 only)")
    if tilelang_time is not None and output_tilelang is not None:
        tilelang_diff = torch.abs(output_tilelang - output_torch).max().item()
        cuda_for_tl = cuda_time if cuda_time is not None else torch_time
        print(f"  TileLang:   {tilelang_time:8.4f} ms  (speedup: {torch_time/tilelang_time:.2f}x, vs CUDA: {cuda_for_tl/tilelang_time:.2f}x)")

    print(f"\nVerification:")
    if cuda_time is not None:
        cuda_diff = torch.abs(output_cuda - output_torch).max().item()
        print(f"  CUDA vs PyTorch max diff: {cuda_diff:.2e}  {'[PASS]' if cuda_diff < 1e-3 else '[FAIL]'}")
    if tilelang_time is not None and output_tilelang is not None:
        tilelang_diff = torch.abs(output_tilelang - output_torch).max().item()
        print(f"  TileLang vs PyTorch max diff: {tilelang_diff:.2e}  {'[PASS]' if tilelang_diff < 1e-2 else '[FAIL]'}")


# =============================================================================
# Test Suite
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Softmax Performance Benchmark: CUDA vs PyTorch vs TileLang")
    print("="*70)

    test_cases = [
        # Small matrices
        (1024, 1024, torch.float16),
        (1024, 1024, torch.float32),

        # Large square matrices
        (4096, 4096, torch.float16),
        (8192, 8192, torch.float16),

        # Wide matrices (many columns)
        (1024, 16384, torch.float16),

        # Large row (attention-like)
        (1, 1024 * 1024, torch.float16),
    ]

    for rows, cols, dtype in test_cases:
        run_benchmark(rows, cols, dtype=dtype)
