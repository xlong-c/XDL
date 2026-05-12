#pragma once
// Torch bridge: 将 torch::Tensor 桥接到 nunchaku kernel

#include <torch/extension.h>
#include <cuda_runtime_api.h>

namespace nunchaku_bridge {

// ============================================================
// W4A4: INT4 weight × INT4 activation GEMM
// act:  [M, K]      — FP16/BF16 输入 (会被 kernel 当作 packed INT4 读取)
// wgt:  [N, K]      — FP16/BF16 输入 (会被 kernel 当作 packed INT4 读取)
// out:  [M, N]      — FP16/BF16 输出
// ascales: [K/64, M] — per-group activation scales
// wscales: [K/64, N] — per-group weight scales
// ============================================================
void gemm_w4a4(torch::Tensor act, torch::Tensor wgt, torch::Tensor out,
               torch::Tensor ascales, torch::Tensor wscales,
               torch::Tensor bias = {});

// ============================================================
// W8A8: INT8 weight × INT8 activation GEMM
// act:  [M, K]      — INT8 输入
// wgt:  [N, K]      — INT8 输入
// out:  [M, N]      — BF16 输出
// ascales: [M]       — per-row activation scales
// wscales: [N]       — per-column weight scales
// ============================================================
void gemm_w8a8(torch::Tensor act, torch::Tensor wgt, torch::Tensor out,
               torch::Tensor ascales, torch::Tensor wscales,
               torch::Tensor bias = {});

// ============================================================
// Flash Attention (FP16)
// q: [Batch, Head, TokensQ, HeadDim] — FP16 packed
// k: [Batch, Head, TokensKV, HeadDim] — FP16 packed
// v: [Batch, Head, TokensKV, HeadDim] — FP16 packed
// o: [Batch, TokensQ, Head * HeadDim] — FP16/BF16 输出
// ============================================================
void attention_fp16(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                    torch::Tensor o, float scale);

// ============================================================
// 量化: FP16/BF16 → 4-bit packed
// input:  [M, K] — FP16/BF16
// output: [M, K/2] — INT8 (packed INT4)
// oscales: [K/64, M] — per-group scales
// ============================================================
void quantize_w4a4_act(torch::Tensor input, torch::Tensor output, torch::Tensor oscales);
void quantize_w4a4_wgt(torch::Tensor input, torch::Tensor output, torch::Tensor oscales);

// ============================================================
// 量化: FP16/BF16 → 8-bit packed
// input:  [M, K] — FP16/BF16
// output: [M, K] — INT8
// oscales: [M] — per-row scales
// ============================================================
void quantize_w8a8_act(torch::Tensor input, torch::Tensor output, torch::Tensor oscales, bool fuse_glu = false);

}  // namespace nunchaku_bridge
