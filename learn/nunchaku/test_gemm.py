#!/usr/bin/env python3
"""
test_gemm.py — 测试 nunchaku GEMM + Attention kernel 的 torch 绑定

使用 W4A4 和 W8A8 量化 GEMM 做矩阵乘法，验证输出正确性。
"""

import torch
import time

# 导入编译好的扩展
try:
    import nunchaku_gemm
    HAS_EXT = True
except ImportError:
    print("[WARN] nunchaku_gemm 扩展未安装，仅做 API 演示")
    print("       运行: cd learn/nunchaku && python setup.py install")
    HAS_EXT = False


def test_w4a4_gemm():
    """测试 W4A4 GEMM: INT4 weight × INT4 activation"""
    if not HAS_EXT:
        print("[SKIP] W4A4 GEMM 测试 — 扩展未安装")
        return

    M, N, K = 512, 256, 1024
    group_size = 64  # W4A4 默认 group size

    print(f"\n=== W4A4 GEMM: M={M}, N={N}, K={K}, group_size={group_size} ===")

    # 1. 准备 FP16 输入
    act_fp16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    wgt_fp16 = torch.randn(N, K, dtype=torch.float16, device="cuda")

    # 2. 量化 activation → INT4 packed
    # 输出: [M, K/2] int8 (2×INT4 打包为 1×INT8)
    act_q = torch.empty(M, K // 2, dtype=torch.int8, device="cuda")
    # 缩放因子: [K/group_size, M] = [K/64, M]
    act_scales = torch.empty(K // group_size, M, dtype=torch.float16, device="cuda")
    nunchaku_gemm.quantize_w4a4_act(act_fp16, act_q, act_scales)

    # 3. 量化 weight → INT4 packed
    wgt_q = torch.empty(N, K // 2, dtype=torch.int8, device="cuda")
    wgt_scales = torch.empty(K // group_size, N, dtype=torch.float16, device="cuda")
    nunchaku_gemm.quantize_w4a4_wgt(wgt_fp16, wgt_q, wgt_scales)

    # 4. W4A4 GEMM
    out = torch.empty(M, N, dtype=torch.float16, device="cuda")
    nunchaku_gemm.gemm_w4a4(act_q, wgt_q, out, act_scales, wgt_scales)

    # 5. 参考: FP16 GEMM (torch)
    ref = act_fp16 @ wgt_fp16.T

    # 6. 比较
    cos_sim = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    )
    mae = (out.float() - ref.float()).abs().mean()
    print(f"  Cosine similarity: {cos_sim.item():.6f}")
    print(f"  MAE: {mae.item():.6f}")


def test_w8a8_gemm():
    """测试 W8A8 GEMM: INT8 weight × INT8 activation"""
    if not HAS_EXT:
        print("[SKIP] W8A8 GEMM 测试 — 扩展未安装")
        return

    M, N, K = 512, 256, 1024

    print(f"\n=== W8A8 GEMM: M={M}, N={N}, K={K} ===")

    # 1. 准备 BF16 输入 (W8A8 使用 BF16)
    act_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    wgt_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # 2. 量化 activation → INT8
    act_q = torch.empty(M, K, dtype=torch.int8, device="cuda")
    act_scales = torch.empty(M, dtype=torch.bfloat16, device="cuda")  # per-row
    nunchaku_gemm.quantize_w8a8_act(act_bf16, act_q, act_scales)

    # 3. 量化 weight → INT8 (需要转换: N×K → N×K)
    # W8A8 weight 需要预量化，这里用简单的 per-tensor 方式
    wgt_q = torch.empty(N, K, dtype=torch.int8, device="cuda")
    wgt_scales = torch.empty(N, dtype=torch.bfloat16, device="cuda")
    # 临时: 直接对 weight 做 per-row 量化
    wgt_absmax = wgt_bf16.abs().max(dim=1).values
    wgt_scales.copy_(wgt_absmax / 127.0)
    wgt_q.copy_((wgt_bf16.float() / (wgt_absmax.float() / 127.0 + 1e-8))
                .round().clamp(-128, 127).to(torch.int8))

    # 4. W8A8 GEMM
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    nunchaku_gemm.gemm_w8a8(act_q, wgt_q, out, act_scales, wgt_scales)

    # 5. 参考
    ref = act_bf16 @ wgt_bf16.T

    # 6. 比较
    cos_sim = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    )
    mae = (out.float() - ref.float()).abs().mean()
    print(f"  Cosine similarity: {cos_sim.item():.6f}")
    print(f"  MAE: {mae.item():.6f}")


def test_attention():
    """测试 Flash Attention"""
    if not HAS_EXT:
        print("[SKIP] Attention 测试 — 扩展未安装")
        return

    B, H, NQ, NKV, D = 2, 8, 256, 256, 128

    print(f"\n=== Flash Attention: B={B}, H={H}, Q={NQ}, KV={NKV}, D={D} ===")

    q = torch.randn(B, H, NQ, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, H, NKV, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, H, NKV, D, dtype=torch.float16, device="cuda")
    o = torch.empty(B, NQ, H * D, dtype=torch.float16, device="cuda")

    scale = D ** -0.5

    nunchaku_gemm.attention_fp16(q, k, v, o, scale)

    # 参考: PyTorch 标准 attention
    q_ref = q.permute(0, 2, 1, 3).reshape(B, NQ, H, D)
    k_ref = k.permute(0, 2, 1, 3).reshape(B, NKV, H, D)
    v_ref = v.permute(0, 2, 1, 3).reshape(B, NKV, H, D)
    attn = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=scale)
    o_ref = attn.reshape(B, NQ, H * D)

    cos_sim = torch.nn.functional.cosine_similarity(
        o.float().flatten(), o_ref.float().flatten(), dim=0
    )
    print(f"  Cosine similarity: {cos_sim.item():.6f}")


def bench_w4a4():
    """性能对比: W4A4 vs FP16"""
    if not HAS_EXT:
        return

    M, N, K = 2048, 4096, 4096
    group_size = 64

    print(f"\n=== 性能对比: W4A4 vs FP16 (M={M}, N={N}, K={K}) ===")

    act_fp16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    wgt_fp16 = torch.randn(N, K, dtype=torch.float16, device="cuda")

    # 量化
    act_q = torch.empty(M, K // 2, dtype=torch.int8, device="cuda")
    act_scales = torch.empty(K // group_size, M, dtype=torch.float16, device="cuda")
    wgt_q = torch.empty(N, K // 2, dtype=torch.int8, device="cuda")
    wgt_scales = torch.empty(K // group_size, N, dtype=torch.float16, device="cuda")

    nunchaku_gemm.quantize_w4a4_act(act_fp16, act_q, act_scales)
    nunchaku_gemm.quantize_w4a4_wgt(wgt_fp16, wgt_q, wgt_scales)

    # Warmup
    out = torch.empty(M, N, dtype=torch.float16, device="cuda")
    for _ in range(10):
        nunchaku_gemm.gemm_w4a4(act_q, wgt_q, out, act_scales, wgt_scales)
    torch.cuda.synchronize()

    # W4A4 timing
    t0 = time.time()
    for _ in range(100):
        nunchaku_gemm.gemm_w4a4(act_q, wgt_q, out, act_scales, wgt_scales)
    torch.cuda.synchronize()
    t_w4a4 = (time.time() - t0) / 100 * 1000

    # FP16 timing
    for _ in range(10):
        ref = act_fp16 @ wgt_fp16.T
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        ref = act_fp16 @ wgt_fp16.T
    torch.cuda.synchronize()
    t_fp16 = (time.time() - t0) / 100 * 1000

    print(f"  W4A4 (nunchaku): {t_w4a4:.3f} ms")
    print(f"  FP16  (torch):    {t_fp16:.3f} ms")
    print(f"  Speedup:          {t_fp16 / t_w4a4:.2f}x")

    # 显存对比
    mem_w4a4 = act_q.element_size() * act_q.numel() + wgt_q.element_size() * wgt_q.numel()
    mem_fp16 = act_fp16.element_size() * act_fp16.numel() + wgt_fp16.element_size() * wgt_fp16.numel()
    print(f"  W4A4 memory: {mem_w4a4 / 1024**2:.1f} MB")
    print(f"  FP16 memory: {mem_fp16 / 1024**2:.1f} MB")
    print(f"  Compression: {mem_fp16 / mem_w4a4:.1f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Nunchaku GEMM Kernel 测试 (torch 绑定)")
    print("=" * 60)

    if not HAS_EXT:
        print("\n扩展未安装。运行以下命令编译：")
        print("  cd learn/nunchaku && python setup.py install")
        print("\n文件列表：")
        import os
        for f in sorted(os.listdir(os.path.dirname(__file__))):
            print(f"  {f}")
    else:
        test_w4a4_gemm()
        test_w8a8_gemm()
        test_attention()
        bench_w4a4()

    print("\n" + "=" * 60)
    print("测试完成")
