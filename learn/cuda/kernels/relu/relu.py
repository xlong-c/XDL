"""
ReLU Performance Benchmark: CUDA vs PyTorch

Tests ReLU implementations across different frameworks:
- CUDA: Custom optimized kernels with various vectorization levels (f32, f32x4, f16, f16x2, f16x8, bf16)
- PyTorch: Native torch.nn.functional.relu
"""

import os
import time
from functools import partial
from types import ModuleType
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Get the current directory and load CUDA extension
curr_dir = os.path.dirname(os.path.abspath(__file__))
cuda_file = os.path.join(curr_dir, "relu.cu")
build_dir = os.path.join(curr_dir, "build")
os.makedirs(build_dir, exist_ok=True)

# Load the CUDA kernel as a python module
lib: Any = load(
    name="relu_lib",
    sources=[cuda_file],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
    build_directory=build_dir,
)


def run_benchmark(
    perf_func: Callable[..., Union[torch.Tensor, None]],
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
    has_out_param: bool = True,
):
    """Benchmark a ReLU implementation."""
    if out is not None:
        out.fill_(0)

    if out is not None:
        for i in range(warmup):
            if has_out_param:
                perf_func(x, out)
            else:
                _ = perf_func(x)
    else:
        for i in range(warmup):
            _ = perf_func(x)

    torch.cuda.synchronize()

    start = time.time()
    if out is not None:
        for i in range(iters):
            if has_out_param:
                perf_func(x, out)
            else:
                _ = perf_func(x)
    else:
        for i in range(iters):
            out = perf_func(x)

    torch.cuda.synchronize()
    end = time.time()

    total_time = (end - start) * 1000
    mean_time = total_time / iters

    out_info = f"out_{tag}"
    assert out is not None, "out should not be None at this point"
    out_cpu = out.flatten().detach().cpu()
    if out_cpu.dtype == torch.bfloat16:
        out_cpu = out_cpu.float()
    out_val = out_cpu.numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")

    if show_all:
        print(out)

    return out, mean_time


def verify_results(output_cuda: torch.Tensor, output_torch: torch.Tensor, tolerance: float = 1e-3):
    """Verify that CUDA and PyTorch outputs match."""
    max_diff = torch.abs(output_cuda - output_torch).max().item()
    passed = max_diff < tolerance
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  Max diff: {max_diff:.2e}  {status}")
    return passed


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

print("=" * 85)
print("ReLU Performance Benchmark: CUDA vs PyTorch")
print("=" * 85)

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")

    print(f"\nFP32 Tests:")
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()

    out_f32, time_f32 = run_benchmark(lib.relu_f32, x, "f32", y)
    out_f32x4, time_f32x4 = run_benchmark(lib.relu_f32x4, x, "f32x4", y)

    def pytorch_relu_with_out(x, out):
        out.copy_(torch.relu(x))

    out_torch, time_torch = run_benchmark(
        lambda x: pytorch_relu_with_out(x, y), x, "f32_th", y, has_out_param=False
    )

    print(f"\nVerification:")
    verify_results(out_f32, out_torch, tolerance=1e-3)
    verify_results(out_f32x4, out_torch, tolerance=1e-3)

    print("-" * 85)

    print(f"\nFP16 Tests:")
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()

    out_f16, time_f16 = run_benchmark(lib.relu_f16, x_f16, "f16", y_f16)
    out_f16x2, time_f16x2 = run_benchmark(
        lib.relu_f16x2, x_f16, "f16x2", y_f16)
    out_f16x8, time_f16x8 = run_benchmark(
        lib.relu_f16x8, x_f16, "f16x8", y_f16)
    out_f16x8_pack, time_f16x8_pack = run_benchmark(
        lib.relu_f16x8_pack, x_f16, "f16x8pack", y_f16
    )

    out_torch_f16, time_torch_f16 = run_benchmark(
        lambda x: pytorch_relu_with_out(x, y_f16), x_f16, "f16_th", y_f16, has_out_param=False
    )

    print(f"\nVerification:")
    verify_results(out_f16, out_torch_f16, tolerance=1e-2)
    verify_results(out_f16x2, out_torch_f16, tolerance=1e-2)
    verify_results(out_f16x8, out_torch_f16, tolerance=1e-2)
    verify_results(out_f16x8_pack, out_torch_f16, tolerance=1e-2)

    print("-" * 85)

    if torch.cuda.is_bf16_supported():
        print(f"\nBF16 Tests:")
        x_bf16 = x.to(torch.bfloat16).contiguous()
        y_bf16 = y.to(torch.bfloat16).contiguous()

        out_bf16, time_bf16 = run_benchmark(
            lib.relu_bf16, x_bf16, "bf16", y_bf16)
        out_bf16x2, time_bf16x2 = run_benchmark(
            lib.relu_bf16x2, x_bf16, "bf16x2", y_bf16
        )
        out_torch_bf16, time_torch_bf16 = run_benchmark(
            lambda x: pytorch_relu_with_out(x, y_bf16), x_bf16, "bf16_th", y_bf16, has_out_param=False
        )

        print(f"\nVerification:")
        verify_results(out_bf16, out_torch_bf16, tolerance=1e-2)
        verify_results(out_bf16x2, out_torch_bf16, tolerance=1e-2)

        print("-" * 85)

print("=" * 85)
print("Benchmark Complete!")
print("=" * 85)
