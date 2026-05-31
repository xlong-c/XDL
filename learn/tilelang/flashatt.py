import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T

tilelang.set_log_level("WARNING")

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}


@dataclass(frozen=True)
class FlashAttConfig:
    batch: int = 4
    heads: int = 16
    seq_q: int = 1024
    seq_kv: int = 1024
    head_dim: int = 64
    causal: bool = False
    dtype: torch.dtype = torch.float16
    block_m: int = 64
    block_n: int = 64
    num_stages: int = 2
    threads: int = 128
    warmup: int = 20
    iters: int = 100
    check_rtol: float = 1e-2
    check_atol: float = 1e-2


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")


def require_sdpa() -> None:
    if not hasattr(F, "scaled_dot_product_attention"):
        raise RuntimeError("当前 PyTorch 版本不支持 scaled_dot_product_attention")


def build_tilelang_flashatt(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    causal: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
):
    if seq_kv < seq_q:
        raise ValueError("当前教学脚本要求 seq_kv >= seq_q")

    scale = (1.0 / head_dim) ** 0.5 * 1.44269504
    q_shape = [batch, heads, seq_q, head_dim]
    kv_shape = [batch, heads, seq_kv, head_dim]
    dtype = T.float16
    accum_dtype = T.float32
    past_len = seq_kv - seq_q

    @tilelang.jit(
        out_idx=[3],
        pass_configs=PASS_CONFIGS,
    )
    def flashatt():
        @T.prim_func
        def main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            out: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(seq_q, block_m),
                heads,
                batch,
                threads=threads,
            ) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                v_shared = T.alloc_shared([block_n, head_dim], dtype)
                o_shared = T.alloc_shared([block_m, head_dim], dtype)

                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, head_dim], accum_dtype)

                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.copy(q[bz, by, bx * block_m : (bx + 1) * block_m, :], q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(
                        T.ceildiv(seq_kv, block_n),
                        T.ceildiv((bx + 1) * block_m + past_len, block_n),
                    )
                    if causal
                    else T.ceildiv(seq_kv, block_n)
                )

                for kv_tile in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        k[bz, by, kv_tile * block_n : (kv_tile + 1) * block_n, :],
                        k_shared,
                    )

                    if causal:
                        for i, j in T.Parallel(block_m, block_n):
                            q_idx = bx * block_m + i + past_len
                            k_idx = kv_tile * block_n + j
                            acc_s[i, j] = T.if_then_else(
                                q_idx >= k_idx,
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.if_then_else(
                                kv_tile * block_n + j >= seq_kv,
                                -T.infinity(acc_s.dtype),
                                0,
                            )

                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale
                        )

                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale
                        )

                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_m, head_dim):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(
                        v[bz, by, kv_tile * block_n : (kv_tile + 1) * block_n, :],
                        v_shared,
                    )
                    T.gemm(
                        acc_s_cast,
                        v_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_m, head_dim):
                    acc_o[i, j] /= logsum[i]

                T.copy(acc_o, o_shared)
                T.copy(o_shared, out[bz, by, bx * block_m : (bx + 1) * block_m, :])

        return main

    return flashatt()


def sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    require_sdpa()
    if causal and q.size(-2) != k.size(-2):
        try:
            from torch.nn.attention.bias import causal_lower_right
        except Exception as exc:
            raise RuntimeError(
                "非方阵 causal 对比需要 torch.nn.attention.bias.causal_lower_right"
            ) from exc

        attn_mask = causal_lower_right(q.size(-2), k.size(-2))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def theoretical_tflops(config: FlashAttConfig, latency_ms: float) -> float:
    flops_per_matmul = (
        2.0
        * config.batch
        * config.heads
        * config.seq_q
        * config.seq_kv
        * config.head_dim
    )
    total_flops = flops_per_matmul * 2.0
    if config.causal:
        total_flops *= 0.5
    return total_flops / (latency_ms * 1e-3) / 1e12


def run_case(config: FlashAttConfig) -> None:
    require_cuda()
    require_sdpa()
    if config.dtype != torch.float16:
        raise ValueError("当前 TileLang 示例固定使用 float16")

    device = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(
        config.batch,
        config.heads,
        config.seq_q,
        config.head_dim,
        device=device,
        dtype=config.dtype,
    ).contiguous()
    k = torch.randn(
        config.batch,
        config.heads,
        config.seq_kv,
        config.head_dim,
        device=device,
        dtype=config.dtype,
    ).contiguous()
    v = torch.randn_like(k).contiguous()

    kernel = build_tilelang_flashatt(
        batch=config.batch,
        heads=config.heads,
        seq_q=config.seq_q,
        seq_kv=config.seq_kv,
        head_dim=config.head_dim,
        causal=config.causal,
        block_m=config.block_m,
        block_n=config.block_n,
        num_stages=config.num_stages,
        threads=config.threads,
    )

    out_tilelang = kernel(q, k, v)
    out_sdpa = sdpa_reference(q, k, v, causal=config.causal)

    torch.testing.assert_close(
        out_tilelang.float(),
        out_sdpa.float(),
        rtol=config.check_rtol,
        atol=config.check_atol,
    )

    latency_sdpa = bench(
        lambda: sdpa_reference(q, k, v, causal=config.causal),
        warmup=config.warmup,
        iters=config.iters,
    )
    latency_tilelang = bench(
        lambda: kernel(q, k, v),
        warmup=config.warmup,
        iters=config.iters,
    )

    print("=" * 100)
    print(
        f"device={torch.cuda.get_device_name(0)}, torch={torch.__version__}, "
        f"tilelang={getattr(tilelang, '__version__', 'unknown')}"
    )
    print(
        f"shape=(B={config.batch}, H={config.heads}, Q={config.seq_q}, "
        f"KV={config.seq_kv}, D={config.head_dim}), causal={config.causal}"
    )
    print(
        f"tile=(block_m={config.block_m}, block_n={config.block_n}, "
        f"stages={config.num_stages}, threads={config.threads})"
    )
    print(f"max diff vs sdpa: {max_diff(out_tilelang, out_sdpa):.6e}")
    print(
        f"latency ms: sdpa={latency_sdpa:.4f}, tilelang={latency_tilelang:.4f}, "
        f"tilelang/sdpa={latency_tilelang / latency_sdpa:.2f}x"
    )
    print(
        f"throughput: sdpa={theoretical_tflops(config, latency_sdpa):.2f} TFLOPS, "
        f"tilelang={theoretical_tflops(config, latency_tilelang):.2f} TFLOPS"
    )


def main() -> None:
    config = FlashAttConfig(
        batch=4,
        heads=16,
        seq_q=1024,
        seq_kv=1024,
        head_dim=64,
        causal=False,
        dtype=torch.float16,
        block_m=64,
        block_n=64,
        num_stages=2,
        threads=128,
        warmup=20,
        iters=100,
        check_rtol=1e-2,
        check_atol=1e-2,
    )
    run_case(config)


if __name__ == "__main__":
    main()
