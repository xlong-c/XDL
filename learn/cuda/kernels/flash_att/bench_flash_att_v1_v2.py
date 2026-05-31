import os
import time
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(CURR_DIR, "build")
os.makedirs(BUILD_DIR, exist_ok=True)

lib = load(
    name="flash_att_v1_v2_lib",
    sources=[
        os.path.join(CURR_DIR, "flash_att_bindings.cu"),
        os.path.join(CURR_DIR, "fla_tiling_qkv_swizzle_qkv_f32f16f16f32.cu"),
        os.path.join(CURR_DIR, "fla_tiling_qkv_swizzle_qkv_f32f16f16f32_v2.cu"),
        os.path.join(CURR_DIR, "fla_tiling_qkv_swizzle_qkv_f32f16f16f32_v3.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
        "-arch=sm_89",
    ],
    extra_cflags=["-std=c++17"],
    build_directory=BUILD_DIR,
)


def bench(
    fn: Callable[[], torch.Tensor | None],
    warmup: int = 20,
    iters: int = 100,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def run_case(
    batch: int,
    heads: int,
    seqlen: int,
    head_dim: int,
    stages: int = 2,
) -> None:
    print("-" * 100)
    print(
        f"B={batch}, H={heads}, N={seqlen}, D={head_dim}, stages={stages}, "
        f"device={torch.cuda.get_device_name(0)}"
    )

    q = torch.randn(
        batch, heads, seqlen, head_dim, device="cuda", dtype=torch.float16
    ).contiguous()
    k = torch.randn_like(q).contiguous()
    v = torch.randn_like(q).contiguous()

    # SDPA expects [B, H, N, D] as well.
    out_sdpa = F.scaled_dot_product_attention(q, k, v)

    out_v1 = torch.empty_like(q)
    out_v2 = torch.empty_like(q)
    out_v3 = torch.empty_like(q)

    lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv(
        q, k, v, out_v1, stages
    )
    lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_v2(
        q, k, v, out_v2, stages
    )
    lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_v3(
        q, k, v, out_v3, stages
    )
    torch.cuda.synchronize()

    print(
        f"max diff vs sdpa: "
        f"v1={max_diff(out_v1, out_sdpa):.6e}, "
        f"v2={max_diff(out_v2, out_sdpa):.6e}, "
        f"v3={max_diff(out_v3, out_sdpa):.6e}, "
        f"v1_vs_v2={max_diff(out_v1, out_v2):.6e}, "
        f"v2_vs_v3={max_diff(out_v2, out_v3):.6e}"
    )

    t_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k, v))
    t_v1 = bench(
        lambda: lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv(
            q, k, v, out_v1, stages
        )
    )
    t_v2 = bench(
        lambda: lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_v2(
            q, k, v, out_v2, stages
        )
    )
    t_v3 = bench(
        lambda: lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_v3(
            q, k, v, out_v3, stages
        )
    )

    print(
        f"time ms: sdpa={t_sdpa:.4f}, v1={t_v1:.4f}, v2={t_v2:.4f}, v3={t_v3:.4f}, "
        f"v1/sdpa={t_v1 / t_sdpa:.2f}x, v2/sdpa={t_v2 / t_sdpa:.2f}x, "
        f"v3/sdpa={t_v3 / t_sdpa:.2f}x, v2/v1={t_v2 / t_v1:.2f}x, "
        f"v3/v2={t_v3 / t_v2:.2f}x"
    )


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())
    print()

    cases = [
        (2, 16, 1024, 64, 2),
        (16, 16, 1024, 128, 2),
        (16, 16, 2048, 64, 2),
        (16, 16, 2048, 128, 2),
        (16, 16, 1024, 256, 2),
    ]
    for case in cases:
        run_case(*case)


if __name__ == "__main__":
    main()
