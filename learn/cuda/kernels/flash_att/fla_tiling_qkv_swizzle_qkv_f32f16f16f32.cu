#include "utils.h"

// 这是一个基于 Tensor Core 与 MMA PTX 的 FlashAttention-2 实验实现。
// 输入 Q/K/V 与输出 O 的形状均为：
//   [batch_size, num_heads, seq_len, head_dim]
// 一个 thread block 负责一个 Q_tile[Br, d]，并遍历同一 head 下的全部 K/V tile。

// FlashAttention-2 论文：
// https://arxiv.org/pdf/2307.08691

// 设计思路：
// 1. 在 QK^T 阶段把 Q 按 warp 切开；
// 2. K/V 对所有 warp 共享，减少 warp 之间通过 SMEM 和 shuffle 的通信；
// 3. 用细粒度 MMA tiling 把 SRAM 开销控制在较小范围内。

// 典型布局 1：MMA = m16n8k16，Br=16x4=64，Bc=8x8=64，共 4 个 warp
// |   64x64   |      warp_KV 0       |
// | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x8) |

// 典型布局 2：MMA = m16n8k16，Br=16x8=128，Bc=8x16=128，共 8 个 warp
// |  128x128  |      warp_KV 0        |
// | warp_QP 0 | MMA 0 ... MMA 0 (x16) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x16) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x16) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x16) |
// | warp_QP 4 | MMA 4 ... MMA 4 (x16) |
// | warp_QP 5 | MMA 5 ... MMA 5 (x16) |
// | warp_QP 6 | MMA 6 ... MMA 6 (x16) |
// | warp_QP 7 | MMA 7 ... MMA 7 (x16) |

// 典型布局 3：MMA = m16n8k16，Br=16x8=128，Bc=8x8=64，共 8 个 warp
// |  128x64  |      warp_KV 0        |
// | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
// | warp_QP 4 | MMA 4 ... MMA 4 (x8) |
// | warp_QP 5 | MMA 5 ... MMA 5 (x8) |
// | warp_QP 6 | MMA 6 ... MMA 6 (x8) |
// | warp_QP 7 | MMA 7 ... MMA 7 (x8) |

// 通过手工调整 shared memory 中的列布局，降低 ldmatrix 读取时的 bank
// 冲突。这里改变的是 SMEM 中的物理摆放方式，不改变矩阵的数学含义。

// i: 行号，j: 原始列号。
// 返回值：元素在 SMEM 中实际使用的列号。
template <const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
  // 以 kStep 个元素为一个列块，把相邻 4 行分成一组。
  // 同一组内保持原顺序；不同组之间用异或方式交错列块位置。
  //
  // 以 kColStride=16, kStep=8 为例：
  //   行 0~3  : 列块顺序仍是 (0, 8)
  //   行 4~7  : 列块顺序变成 (8, 0)
  //   行 8~11 : 列块顺序仍是 (0, 8)
  //   行 12~15: 列块顺序变成 (8, 0)
  //
  // 这样做的目的，是让 warp 内线程访问 SMEM 时更分散，减少 bank
  // 冲突，尤其适合后续 ldmatrix 的读取模式。
  //
  // swizzle 公式：
  //   ((j / kStep) ^ (i / 4)) % (kColStride / kStep) * kStep
  static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
  static_assert(kColStride % kStep == 0,
                "kColStride must be multiple of kStep.");
  if constexpr (kStep == 8) {
    return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
  } else {
    static_assert(kStep == 4);
    return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
  }
}

// Q 在 SMEM 中使用同样的 swizzle 规则。
// 这里通常按 8 个 half 为一组搬运，对应 16B 对齐访问。
template <const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_Q_j(int i, int j) {
  return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}

// K 在 SMEM 中使用同样的 swizzle 规则。
template <const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_K_j(int i, int j) {
  return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}

// V 在 SMEM 中使用同样的 swizzle 规则。
// 这样从 SMEM 读 V 片段到寄存器时，ldmatrix.x2.trans 的访存冲突更少。
template <const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_V_j(int i, int j) {
  return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}

// 在 MMA 级别做细粒度 tiling 后，Q@K^T 与 P@V 都只需要保存当前正在计算的
// 小块。这样 Q/K/V 的片上存储开销近似与 Br * 16 同阶，而不是与整段 seqlen
// 同阶。

template <
    const int kHeadDim,            // head_dim，例如 32/64/128
    const int kMmaAtomM,           // 单个 MMA 原子块在 M 方向的大小，固定 16
    const int kMmaAtomN,           // 单个 MMA 原子块在 N 方向的大小，固定 8
    const int kMmaAtomK,           // 单个 MMA 原子块在 K 方向的大小，固定 16
    const int kMmaTileSeqLenQ,     // QK^T 阶段，block 在 M 方向串接多少个 MMA 子块
    const int kMmaTileSeqLenK,     // QK^T 阶段，block 在 N 方向串接多少个 MMA 子块
    const int kMmaTileSeqLenP,     // P@V 阶段，block 在 M 方向串接多少个 MMA 子块
    const int kMmaTileHeadDimV,    // P@V 阶段，block 在 N 方向串接多少个 MMA 子块
    const int kWarpTileSeqLenQ,    // 单个 warp 在 QK^T 阶段覆盖多少个 M 子块
    const int kWarpTileSeqLenK,    // 单个 warp 在 QK^T 阶段覆盖多少个 N 子块
    const int kWarpTileSeqLenP,    // 单个 warp 在 P@V 阶段覆盖多少个 M 子块
    const int kWarpTileHeadDimV,   // 单个 warp 在 P@V 阶段覆盖多少个 head_dim 子块
    const int kOStorageAccFloat32, // 最终 O 在寄存器中按 fp32 还是 half 暂存
    const int kStage,              // cp.async 流水 stage 数，当前支持 1 或 2
    const int kPadQ,               // Q 的 SMEM 行补齐大小，单位：half
    const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE *kMmaTileSeqLenQ *kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_kernel(
        half *Q, half *K, half *V, half *O, int QKV_seqlen, int QKV_head) {
  // 计算布局：
  //   Q[Br, d] @ K^T[d, Bc]
  //   P[Br, Bc] @ V[Bc, d]
  // 其中 K 在内存里是行主序的 [Bc, d]，但参与计算时等价于按列主序读取 K^T[d, Bc]。
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 &&
                kMmaAtomK == 16);                                 // 当前只支持 m16n8k16
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);    // QK^T 阶段的 block 布局限制
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);   // P@V 阶段的 block 布局限制
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16); // QK^T 阶段的 warp 布局限制
  // kWarpTileHeadDimV 决定一个 warp 覆盖多少个 8 列宽的 V 子块。
  // 例如：
  //   kWarpTileHeadDimV = 8  -> d =  8 * 8  = 64
  //   kWarpTileHeadDimV = 16 -> d = 16 * 8 = 128
  static_assert(kWarpTileSeqLenP == 1 &&
                kWarpTileHeadDimV ==
                    (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV))); // P@V 阶段的维度必须对齐
  static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
  static_assert(kStage < 3 && kStage > 0);
  static_assert(kPadQ >= 0 && kPadQ % 8 == 0); // 便于 16B 对齐访问
  static_assert(kPadK >= 0 && kPadK % 8 == 0);
  static_assert(kPadV >= 0 && kPadV % 8 == 0);
  constexpr int Br =
      kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // block 在 Q/P 的行方向覆盖范围
  constexpr int Bc =
      kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; // block 在 K/V 的序列方向覆盖范围
  static_assert(Br >= Bc);                            // 便于后面复用 shared memory
  constexpr int kNumThreads =
      WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // block 总线程数
  // Tc 表示沿 seqlen 方向一共要遍历多少个 K/V tile。
  const int Tc = div_ceil(QKV_seqlen, Bc);
  const float scale = 1.0f / sqrt((float)kHeadDim);

  // 网格布局：
  //   x 维按 Q tile 划分；
  //   y 维把 batch 与 head 打平成一个维度。
  const int QKV_batch_id = blockIdx.y / QKV_head; // 当前 batch 下标
  const int QKV_head_id = blockIdx.y % QKV_head;  // 当前 head 下标
  const int Q_tile_id = blockIdx.x;               // 当前处理的 Q tile 编号
  const int O_tile_id = Q_tile_id;                // 输出 O 与 Q 使用相同的 tile 编号
  const int tid = threadIdx.x;                    // block 内线程编号
  const int warp_id = tid / WARP_SIZE;            // block 内 warp 编号
  const int lane_id = tid % WARP_SIZE;            // warp 内 lane 编号
  const int warp_QP = warp_id;                    // 负责 Q/P 行方向计算的 warp 编号
  const int warp_KV = 0;                          // 当前实现里所有 warp 共享同一组 KV 访问模式
  const int Q_gmem_offset =
      ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim)); // Q 在当前 batch/head 下的起始偏移
  const int K_gmem_offset =
      ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim)); // K 在当前 batch/head 下的起始偏移
  const int V_gmem_offset = Q_gmem_offset;     // V 与 Q 形状相同，因此偏移公式一致
  const int O_gmem_offset = Q_gmem_offset;     // O 与 Q 形状相同，因此偏移公式一致

  // 下面几组索引描述 gmem -> 线程 -> smem 的搬运映射关系。
  // Q tile 的形状是 [Br, 16]，每个线程搬一小段连续元素。
  int load_smem_Q_Br = (tid / (kNumThreads / Br)); // Q tile 中的行号
  int load_smem_Q_d =
      (tid % (kNumThreads / Br)) *
      (kMmaAtomK / (kNumThreads / Br)); // Q tile 中的列起点，通常是 0 或 8
  // K tile 的形状是 [Bc, 16]。
  int load_smem_K_Bc = (tid / (kNumThreads / Bc)); // K tile 中的行号
  int load_smem_K_d =
      (tid % (kNumThreads / Bc)) *
      (kMmaAtomK / (kNumThreads / Bc)); // K tile 中的列起点
  // V 每次按 [Bc, 16] 的小块搬到 shared memory。
  int load_smem_V_Bc =
      (tid / (kNumThreads / Bc)); // V tile 中的行号
  int load_smem_V_d =
      (tid % (kNumThreads / Bc)) *
      ((kMmaAtomN * 2) / (kNumThreads / Bc)); // V tile 中的列起点
  // 当前 block 在 Q 中负责的全局行号。
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  if (load_gmem_Q_Br >= QKV_seqlen)
    return;

  // shared memory 只给 Q/K/V 使用。
  // O 的回写通过寄存器复用和 warp shuffle 完成，不额外占用 shared memory。
  extern __shared__ half smem[];
  // 下面这些大小都以 half 元素个数为单位，不是字节数。
  constexpr int Q_tile_size =
      Br * (kMmaAtomK + kPadQ);                             // Q tile 的 shared memory 元素数
  constexpr int K_tile_size = Bc * (kMmaAtomK + kPadK);     // K tile 的 shared memory 元素数
  constexpr int V_tile_size = Bc * (kMmaAtomN * 2 + kPadV); // V tile 的 shared memory 元素数
  half *Q_tile_smem = smem;                                 // Q 从 smem 起始位置开始放
  half *K_tile_smem = Q_tile_smem + kStage * Q_tile_size;   // K 接在 Q 的 stage 缓冲区后面
  half *V_tile_smem = Q_tile_smem;                          // QK^T 完成后，V 复用 Q 那片 smem
  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // 保存 online softmax 的历史状态。
  // m_old: 当前行已经见过的最大 score。
  // l_old: 当前行已经累计的 softmax 分母，即 sum(exp(score - m_old))。
  // 这两个量按 lane 保存，并用 float 保证精度。
  float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  // Tensor Core 所需的寄存器片段。
  // Q/K/V 不是整块矩阵，而是“当前线程参与一次 MMA 需要的那部分片段”。
  uint32_t R_Q[kWarpTileSeqLenQ][4]; // [1][4]
  uint32_t R_K[kWarpTileSeqLenK][2]; // [8][2]
  uint32_t R_V[2];                   // V 作为 MMA 的 B 操作数，每线程只需 2 个 32-bit 寄存器。
  // R_S: 当前 K/V tile 上的分数块 S_tile = Q_tile @ K_tile^T，随后会原地改写成 P。
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][4]; // [1][8][4]，按 fp32 累加
  uint32_t R_O[4];                                     // 当前一次 P@V 的 MMA 累加结果，格式是 fp32 累加器片段
  // R_D: warp 负责的最终输出 O 的寄存器缓存。
  // 第 1 维：seq 方向的 warp tile（本 kernel 中固定为 1）。
  // 第 2 维：head_dim 方向按 8 列一组切开的 tile 编号。
  // 第 3 维：该 tile 在当前线程里占用的寄存器个数；存 fp32 时为 4，存 half 时为 2。
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV]
              [(kOStorageAccFloat32) ? 4 : 2];
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV,
               ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);

  // 沿 seqlen 方向遍历所有 K/V tile。
  // 每次先算一个分数块：
  //   S_tile[Br, Bc] = Q_tile[Br, d] @ K_tile^T[d, Bc]
#pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) {
    // TODO: 最后一个 tile 可能不足完整块；当前实现默认已经按 tile 对齐或外部补齐。

    // 预取本轮将要使用的 Q/K 子块：全局内存 -> shared memory。
    if constexpr (kStage > 1) {
#pragma unroll
      for (int stage = 0; stage < (kStage - 1); ++stage) {
        // 搬运 Q 的一个 [Br, 16] 子块。
        int load_gmem_Q_d = (stage * kMmaAtomK) + load_smem_Q_d;
        int load_gmem_Q_addr =
            (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
#pragma unroll
        for (int i = 0; i < (kMmaAtomK / (kNumThreads / Br)); i += 8) {
          uint32_t load_smem_Q_ptr =
              (smem_Q_base_ptr +
               (stage * Q_tile_size + load_smem_Q_Br * (kMmaAtomK + kPadQ) +
                swizzle_permuted_Q_j<kMmaAtomK>(load_smem_Q_Br,
                                                load_smem_Q_d + i)) *
                   sizeof(half));
          CP_ASYNC_CG(load_smem_Q_ptr, &Q[load_gmem_Q_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

        // 搬运 K 的一个 [Bc, 16] 子块。
        int load_gmem_K_Bc = (tile_K_seqlen * Bc) + load_smem_K_Bc;
        int load_gmem_K_d =
            (stage * kMmaAtomK) + load_smem_K_d;
        int load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
#pragma unroll
        for (int i = 0; i < (kMmaAtomK / (kNumThreads / Bc)); i += 8) {
          uint32_t load_smem_K_ptr =
              (smem_K_base_ptr +
               (stage * K_tile_size + load_smem_K_Bc * (kMmaAtomK + kPadK) +
                swizzle_permuted_K_j<kMmaAtomK>(load_smem_K_Bc,
                                                load_smem_K_d + i)) *
                   sizeof(half));
          CP_ASYNC_CG(load_smem_K_ptr, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }

      // 让最早一组异步拷贝完成，然后再进入计算。
      CP_ASYNC_WAIT_GROUP(kStage - 2); // stage=2 时等待 0 组未完成请求
      __syncthreads();
    }

    // 接下来沿 d 方向遍历 Q/K 的子块，逐步累加出完整的 S_tile。
    // 这里每次处理 16 列，因为 MMA 原子块固定是 k=16。
    fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 4>(R_S, 0);
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      // 当前 d 子块对应的 stage 缓冲区编号。
      int smem_sel = (tile_K_d) % kStage;
      // 下一轮 d 子块将写入的 stage 缓冲区编号。
      int smem_sel_next = (tile_K_d + (kStage - 1)) % kStage;

      // stage>1 时，一边算当前子块，一边预取下一个子块。
      if constexpr (kStage > 1) {
        if ((tile_K_d + 1) < (kHeadDim / kMmaAtomK)) {
          // 预取下一个 Q 子块。
          int load_gmem_Q_d = ((tile_K_d + 1) * kMmaAtomK) + load_smem_Q_d;
          int load_gmem_Q_addr =
              (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
#pragma unroll
          for (int i = 0; i < (kMmaAtomK / (kNumThreads / Br)); i += 8) {
            uint32_t load_smem_Q_ptr =
                (smem_Q_base_ptr + (smem_sel_next * Q_tile_size +
                                    load_smem_Q_Br * (kMmaAtomK + kPadQ) +
                                    swizzle_permuted_Q_j<kMmaAtomK>(
                                        load_smem_Q_Br, load_smem_Q_d + i)) *
                                       sizeof(half));
            CP_ASYNC_CG(load_smem_Q_ptr, &Q[load_gmem_Q_addr + i], 16);
          }
          CP_ASYNC_COMMIT_GROUP();

          // 预取下一个 K 子块。
          int load_gmem_K_Bc = tile_K_seqlen * Bc + load_smem_K_Bc;
          int load_gmem_K_d = ((tile_K_d + 1) * kMmaAtomK) +
                              load_smem_K_d;
          int load_gmem_K_addr =
              (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
#pragma unroll
          for (int i = 0; i < (kMmaAtomK / (kNumThreads / Bc)); i += 8) {
            uint32_t load_smem_K_ptr =
                (smem_K_base_ptr + (smem_sel_next * K_tile_size +
                                    load_smem_K_Bc * (kMmaAtomK + kPadK) +
                                    swizzle_permuted_K_j<kMmaAtomK>(
                                        load_smem_K_Bc, load_smem_K_d + i)) *
                                       sizeof(half));
            CP_ASYNC_CG(load_smem_K_ptr, &K[load_gmem_K_addr + i], 16);
          }
          CP_ASYNC_COMMIT_GROUP();
        }
      } else {
        // 没有流水时，就同步搬运当前 Q/K 子块。
        int load_gmem_Q_d = (tile_K_d * kMmaAtomK) + load_smem_Q_d;
        int load_gmem_Q_addr =
            (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
#pragma unroll
        for (int i = 0; i < (kMmaAtomK / (kNumThreads / Br)); i += 8) {
          uint32_t load_smem_Q_ptr =
              (smem_Q_base_ptr +
               (smem_sel * Q_tile_size + load_smem_Q_Br * (kMmaAtomK + kPadQ) +
                swizzle_permuted_Q_j<kMmaAtomK>(load_smem_Q_Br,
                                                load_smem_Q_d + i)) *
                   sizeof(half));
          CP_ASYNC_CG(load_smem_Q_ptr, &Q[load_gmem_Q_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

        int load_gmem_K_Bc = (tile_K_seqlen * Bc) + load_smem_K_Bc;
        int load_gmem_K_d =
            (tile_K_d * kMmaAtomK) + load_smem_K_d;
        int load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
#pragma unroll
        for (int i = 0; i < (kMmaAtomK / (kNumThreads / Bc)); i += 8) {
          uint32_t load_smem_K_ptr =
              (smem_K_base_ptr +
               (smem_sel * K_tile_size + load_smem_K_Bc * (kMmaAtomK + kPadK) +
                swizzle_permuted_K_j<kMmaAtomK>(load_smem_K_Bc,
                                                load_smem_K_d + i)) *
                   sizeof(half));
          CP_ASYNC_CG(load_smem_K_ptr, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
        // 等待当前 Q/K 子块到位，再从 SMEM 读入寄存器。
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
      }

      // 从 shared memory 读取 Q 片段到寄存器。
      static_assert(kWarpTileSeqLenQ == 1);
      { // kWarpTileSeqLenQ = 1，对应 Q[Br, d] 的一个 warp 子块
        int warp_smem_Q_Br =
            warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + 0 * kMmaAtomM;
        int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16; // 0~15
        int lane_smem_Q_d = (lane_id / 16) * 8;             // 0,8
        uint32_t lane_smem_Q_ptr =
            (smem_Q_base_ptr +
             (smem_sel * Q_tile_size + lane_smem_Q_Br * (kMmaAtomK + kPadQ) +
              swizzle_permuted_Q_j<kMmaAtomK>(lane_smem_Q_Br, lane_smem_Q_d)) *
                 sizeof(half));
        LDMATRIX_X4(R_Q[0][0], R_Q[0][1], R_Q[0][2], R_Q[0][3],
                    lane_smem_Q_ptr);
      }

      // 从 shared memory 读取 K 片段到寄存器。
      // 这里 K 在数学上按 K^T 参与计算，因此读取布局与 Q 不同。
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int warp_smem_K_Bc =
            warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
        int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8; // 0~7
        int lane_smem_K_d = ((lane_id / 8) % 2) * 8;       // 0,8
        uint32_t lane_smem_K_ptr =
            (smem_K_base_ptr +
             (smem_sel * K_tile_size + lane_smem_K_Bc * (kMmaAtomK + kPadK) +
              swizzle_permuted_K_j<kMmaAtomK>(lane_smem_K_Bc, lane_smem_K_d)) *
                 sizeof(half));
        LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr); // R_K
      }
      if constexpr (kStage < 2) {
        // 单 stage 模式下，需要在进入下一轮前同步，避免覆盖仍在使用的缓冲区。
        __syncthreads();
      }

      // 用 Tensor Core 计算当前 d 子块对 S_tile 的贡献，并累加到 R_S。
      static_assert(kWarpTileSeqLenQ == 1);
      { // kWarpTileSeqLenQ = 1
#pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) {
          HMMA16816F32(R_S[0][j][0], R_S[0][j][1], R_S[0][j][2], R_S[0][j][3],
                       R_Q[0][0], R_Q[0][1], R_Q[0][2], R_Q[0][3], R_K[j][0],
                       R_K[j][1], R_S[0][j][0], R_S[0][j][1], R_S[0][j][2],
                       R_S[0][j][3]);
        }
      }

      if constexpr (kStage > 1) {
        // 等待下一轮预取完成，准备切到新的 stage 缓冲区。
        CP_ASYNC_WAIT_GROUP(kStage - 2);
        __syncthreads();
      }

    } // d 方向遍历结束，此时 R_S 中已得到完整的 S_tile
    __syncthreads();

    // 对当前 S_tile 做按行的在线数值稳定 softmax。
    // 这里先求当前 tile 的行最大值和行指数和，后面再与历史的 m_old/l_old 合并。
    float lane_row_max_new[kWarpTileSeqLenQ][2]; // [1][2]
    float lane_row_sum_new[kWarpTileSeqLenQ][2]; // [1][2]
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    // 第一步：求当前 S_tile 的逐行最大值。
    // 先在线程局部求，再在 4-thread 小组内规约。
    { // kWarpTileSeqLenQ = 1
      // 线程内先遍历自己持有的 Bc 方向片段。
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        // R_S 中每个线程持有 4 个 fp32 累加值：
        //   前两个对应一组行，后两个对应另一组行。
        float *t_fptr_S_0_1 = reinterpret_cast<float *>(&(R_S[0][j][0]));
        // 这里乘上 scale，得到 softmax 真正使用的 score = S / sqrt(d)。
        float tmp_max_0 = max(t_fptr_S_0_1[0], t_fptr_S_0_1[1]) * scale;
        float tmp_max_1 = max(t_fptr_S_0_1[2], t_fptr_S_0_1[3]) * scale;
        lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
      } // end for kWarpTileSeqLenK

      // 4 个线程组成一组，继续规约出这两行的最大值。
      lane_row_max_new[0][0] =
          warp_reduce_max<float, 4>(lane_row_max_new[0][0]);
      lane_row_max_new[0][1] =
          warp_reduce_max<float, 4>(lane_row_max_new[0][1]);
    } // end for kWarpTileSeqLenQ

    static_assert(kWarpTileSeqLenQ == 1);
    // 第二步：用新的行最大值把当前 tile 转成 exp(score - m_new)，并求行和。
    { // kWarpTileSeqLenQ = 1
      // 这两个变量分别对应本线程持有的两组行。
      float block_row_max_new_0 = lane_row_max_new[0][0];
      float block_row_max_new_1 = lane_row_max_new[0][1];

      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      // 合并历史最大值，得到截止当前 tile 的最新 m。
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        float *t_fptr_S_0_1 = reinterpret_cast<float *>(&(R_S[0][j][0]));
        half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
        // 先做 score / sqrt(d)，再减去行最大值，最后取 exp。
        // 这样得到的就是当前 tile 的未归一化 softmax 权重 P。
        t_fptr_S_0_1[0] =
            __expf(__fmaf_rn(t_fptr_S_0_1[0], scale, -block_row_max_new_0));
        t_fptr_S_0_1[1] =
            __expf(__fmaf_rn(t_fptr_S_0_1[1], scale, -block_row_max_new_0));
        t_fptr_S_0_1[2] =
            __expf(__fmaf_rn(t_fptr_S_0_1[2], scale, -block_row_max_new_1));
        t_fptr_S_0_1[3] =
            __expf(__fmaf_rn(t_fptr_S_0_1[3], scale, -block_row_max_new_1));
        lane_row_sum_new[0][0] += (t_fptr_S_0_1[0] + t_fptr_S_0_1[1]);
        lane_row_sum_new[0][1] += (t_fptr_S_0_1[2] + t_fptr_S_0_1[3]);
        // R_S 后面要直接作为 P 参与 P@V，所以这里原地改写并转成 half。
        t_hptr_S_0_1[0] = __float2half_rn(t_fptr_S_0_1[0]);
        t_hptr_S_0_1[1] = __float2half_rn(t_fptr_S_0_1[1]);
        t_hptr_S_0_1[2] = __float2half_rn(t_fptr_S_0_1[2]);
        t_hptr_S_0_1[3] = __float2half_rn(t_fptr_S_0_1[3]);
      } // end for kWarpTileSeqLenK

      // 再把每线程的局部和规约成当前 tile 的行指数和 l_new。
      lane_row_sum_new[0][0] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[0][0]);
      lane_row_sum_new[0][1] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[0][1]);
    }

    // 若启用流水，则预取后面要参与 P@V 的 V 子块。
    static_assert(kWarpTileSeqLenP == 1);
    { // kWarpTileSeqLenP = 1
      if constexpr (kStage > 1) {
#pragma unroll
        for (int stage = 0; stage < (kStage - 1); ++stage) {
          int load_gmem_V_Bc =
              (tile_K_seqlen * Bc) + load_smem_V_Bc;
          int load_gmem_V_d = (stage * kMmaAtomN * 2) +
                              load_smem_V_d;
          int load_gmem_V_addr =
              (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
#pragma unroll
          for (int i = 0; i < (kMmaAtomN * 2 / (kNumThreads / Bc)); i += 8) {
            uint32_t load_smem_V_ptr =
                (smem_V_base_ptr + (stage * V_tile_size +
                                    load_smem_V_Bc * (kMmaAtomN * 2 + kPadV) +
                                    swizzle_permuted_V_j<kMmaAtomN * 2>(
                                        load_smem_V_Bc, load_smem_V_d + i)) *
                                       sizeof(half));
            CP_ASYNC_CG(load_smem_V_ptr, &V[load_gmem_V_addr + i], 16);
          }
          CP_ASYNC_COMMIT_GROUP();
        }
      }
    }

    static_assert(kWarpTileSeqLenP == 1);
    {
      // 把当前 tile 的统计量与历史状态合并。
      // 记：
      //   m_old, l_old: 之前所有 tile 的状态
      //   m_new, l_new: 当前 tile 的状态
      // 合并公式：
      //   m = max(m_old, m_new)
      //   l = exp(m_old - m) * l_old + l_new
      float block_row_max_new_0 = lane_row_max_new[0][0];
      float block_row_max_new_1 = lane_row_max_new[0][1];
      float block_row_sum_new_0 = lane_row_sum_new[0][0];
      float block_row_sum_new_1 = lane_row_sum_new[0][1];

      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      // 第一个 tile 时，m_old 初始为 -inf，这里会自动退化成当前 tile 的最大值。
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
      // 第一个 tile 不需要重缩放旧结果，否则 exp(-inf - m) 会带来无意义的数值。
      block_row_max_old_0 =
          (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 =
          (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);
      // 旧的 O 和 l 都是按旧最大值 m_old 保存的。
      // 要和当前 tile 合并，先乘 exp(m_old - m) 把它们换到新的基准上。
      float rescale_o_factor_0 =
          __expf(block_row_max_old_0 - block_row_max_new_0);
      float rescale_o_factor_1 =
          __expf(block_row_max_old_1 - block_row_max_new_1);

      // 等待当前所需的 V 子块预取完成。
      if constexpr (kStage > 1) {
        CP_ASYNC_WAIT_GROUP(kStage - 2); // s2->0, s3->1, s4->2
        __syncthreads();
      }

      // 对 V 的每个 head_dim 子块做 P@V，得到输出 O 的局部结果。
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        // 清空当前这次 MMA 的累加器。
        fill_1D_regs<uint32_t, 4>(R_O, 0);

        int smem_sel_v = (j / 2) % kStage;
        int smem_sel_v_next = ((j / 2) + (kStage - 1)) % kStage;
        // V 每次按 [Bc, 16] 子块搬运；j 每增加 2，就切到下一组 16 列。
        if (j % 2 == 0) {
          if constexpr (kStage > 1) {
            if (((j / 2) + 1) < (kWarpTileHeadDimV / 2)) {
              int load_gmem_V_Bc =
                  (tile_K_seqlen * Bc) + load_smem_V_Bc;
              int load_gmem_V_d = (((j / 2) + 1) * kMmaAtomN * 2) +
                                  load_smem_V_d;
              int load_gmem_V_addr =
                  (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
#pragma unroll
              for (int i = 0; i < (kMmaAtomN * 2 / (kNumThreads / Bc));
                   i += 8) {
                uint32_t load_smem_V_ptr =
                    (smem_V_base_ptr +
                     (smem_sel_v_next * V_tile_size +
                      load_smem_V_Bc * (kMmaAtomN * 2 + kPadV) +
                      swizzle_permuted_V_j<kMmaAtomN * 2>(load_smem_V_Bc,
                                                          load_smem_V_d + i)) *
                         sizeof(half));
                CP_ASYNC_CG(load_smem_V_ptr, &V[load_gmem_V_addr + i], 16);
              }
              CP_ASYNC_COMMIT_GROUP();
            }
          } else {
            // 单 stage 模式下，当前轮现取现用。
            int load_gmem_V_Bc =
                (tile_K_seqlen * Bc) + load_smem_V_Bc;
            int load_gmem_V_d = ((j / 2) * kMmaAtomN * 2) +
                                load_smem_V_d;
            int load_gmem_V_addr =
                (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
#pragma unroll
            for (int i = 0; i < (kMmaAtomN * 2 / (kNumThreads / Bc)); i += 8) {
              uint32_t load_smem_V_ptr =
                  (smem_V_base_ptr + (smem_sel_v * V_tile_size +
                                      load_smem_V_Bc * (kMmaAtomN * 2 + kPadV) +
                                      swizzle_permuted_V_j<kMmaAtomN * 2>(
                                          load_smem_V_Bc, load_smem_V_d + i)) *
                                         sizeof(half));
              CP_ASYNC_CG(load_smem_V_ptr, &V[load_gmem_V_addr + i], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
            // 等待当前 V 子块到位。
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
          }
        }

#pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
          // 从 shared memory 取出当前要参与 P@V 的 V 片段。
          int warp_smem_V_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) +
                              (j % 2) * kMmaAtomN; // 当前 8 列输出所在的 d 偏移
          int lane_smem_V_Bc =
              tile_V_Bc * kMmaAtomK + lane_id % 16; // 当前 16 行 K 维片段中的行号
          int lane_smem_V_d = warp_smem_V_d;
          uint32_t lane_smem_V_ptr =
              (smem_V_base_ptr + (smem_sel_v * V_tile_size +
                                  lane_smem_V_Bc * (kMmaAtomN * 2 + kPadV) +
                                  swizzle_permuted_V_j<kMmaAtomN * 2>(
                                      lane_smem_V_Bc, lane_smem_V_d)) *
                                     sizeof(half));
          LDMATRIX_X2_T(R_V[0], R_V[1], lane_smem_V_ptr); // R_V
          // P 沿 Bc 方向是按 16 列一段被消费的：
          //   tile_V_Bc = 0 -> 取 P[:,  0:16]
          //   tile_V_Bc = 1 -> 取 P[:, 16:32]
          //   tile_V_Bc = 2 -> 取 P[:, 32:48]
          //   tile_V_Bc = 3 -> 取 P[:, 48:64]
          // 每个 16 列片段对应 R_S 中相邻的两个 MMA 片段。
          int w = tile_V_Bc * 2;
          HMMA16816F32(R_O[0], R_O[1], R_O[2], R_O[3], R_S[0][w][0],
                       R_S[0][w][1], R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[0],
                       R_V[1], R_O[0], R_O[1], R_O[2], R_O[3]);
        }
        if constexpr (kStage < 2) {
          // 单 stage 模式下，等当前轮 P@V 完成后再允许覆盖 V 缓冲区。
          __syncthreads();
        }

        // R_O 是当前 tile 贡献的 [Br, 8] 输出片段。
        // R_D 保存的是之前所有 tile 的累计结果。
        // 合并时不能直接相加，而要先把旧结果按新的 m 重缩放：
        //   O = exp(m_old - m) * O_old + (P @ V)_tile
        float *t_fptr_O_0_1 = reinterpret_cast<float *>(&(R_O[0]));
        if constexpr (kOStorageAccFloat32) {
          // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3}
          float *t_fptr_D_0_1 =
              reinterpret_cast<float *>(&(R_D[0][j][0])); // kWarpTileSeqLenP=1
          t_fptr_D_0_1[0] =
              __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[0], t_fptr_O_0_1[0]);
          t_fptr_D_0_1[1] =
              __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[1], t_fptr_O_0_1[1]);
          t_fptr_D_0_1[2] =
              __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[2], t_fptr_O_0_1[2]);
          t_fptr_D_0_1[3] =
              __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[3], t_fptr_O_0_1[3]);
        } else {
          half *t_hptr_D_0_1 =
              reinterpret_cast<half *>(&(R_D[0][j][0])); // kWarpTileSeqLenP=1
          t_hptr_D_0_1[0] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[0]),
                        t_fptr_O_0_1[0]));
          t_hptr_D_0_1[1] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[1]),
                        t_fptr_O_0_1[1]));
          t_hptr_D_0_1[2] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[2]),
                        t_fptr_O_0_1[2]));
          t_hptr_D_0_1[3] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[3]),
                        t_fptr_O_0_1[3]));
        }
        // TODO: 对超大 head_dim，可考虑把已经缩放好的 O 直接分段写回 gmem，
        // 以降低 R_D 占用的寄存器数量。
        if constexpr (kStage > 1) {
          // 等下一组 V 预取完成，再切换到新的 stage 缓冲区。
          CP_ASYNC_WAIT_GROUP(kStage - 2);
          __syncthreads();
        }
      }
      // O 合并完成后，再更新分母 l 和最大值 m。
      float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[0][1];
      // l 对应 softmax 分母的累计值。
      lane_block_row_sum_old[0][0] = (__fmaf_rn(
          rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
      lane_block_row_sum_old[0][1] = (__fmaf_rn(
          rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));
      // 保存最新的 m，供下一个 tile 继续使用。
      lane_block_row_max_old[0][0] = block_row_max_new_0;
      lane_block_row_max_old[0][1] = block_row_max_new_1;
    }
    __syncthreads();
  }
  __syncthreads();

  // 所有 K/V tile 都处理完后，还差最后一步归一化：
  //   O = O / l_final
  // 这里的 l_final 就是整行 softmax 的分母。
  static_assert(kWarpTileSeqLenP == 1);
  {
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      // 在寄存器里做最终缩放；如果 R_D 里存的是 fp32，这里顺便转成 half，
      // 为后面的全局内存回写做准备。
      if constexpr (kOStorageAccFloat32) {
        float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
        half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
        t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[0]);
        t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[1]);
        t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[2]);
        t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[3]);
      } else {
        half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
        t_hptr_D_0_1[0] =
            __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[0]));
        t_hptr_D_0_1[1] =
            __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[1]));
        t_hptr_D_0_1[2] =
            __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[2]));
        t_hptr_D_0_1[3] =
            __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[3]));
      }
    }
  }

  // 最后把寄存器中的 O 回写到全局内存。
  // 这里复用 R_Q/R_K 作为临时打包缓冲区，并借助 warp shuffle 组装连续 128-bit 写。
  static_assert(kWarpTileSeqLenP == 1);
  {
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      // 复用 R_Q/R_K 这两片寄存器作为写回前的整理缓冲区。
      uint32_t *t_uptr_Z_0 = reinterpret_cast<uint32_t *>(&(R_Q[0][0]));
      uint32_t *t_uptr_Z_1 = reinterpret_cast<uint32_t *>(&(R_K[0][0]));
      t_uptr_Z_0[0] = R_D[0][j][0];
      t_uptr_Z_1[0] = R_D[0][j][1];
      t_uptr_Z_0[1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
      t_uptr_Z_0[2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
      t_uptr_Z_0[3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
      t_uptr_Z_1[1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
      t_uptr_Z_1[2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
      t_uptr_Z_1[3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);

      // 每 4 个线程合作组成连续的 128-bit 向量写。
      if (lane_id % 4 == 0) {
        // 当前 lane 对应输出 tile 中的行号。
        int store_warp_regs_O_Br =
            warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
        int store_lane_gmem_O_Br =
            O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
        // 当前 lane 对应输出 tile 中的列号。
        int store_warp_regs_O_d =
            warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        int store_lane_gmem_O_d = store_warp_regs_O_d;
        int store_gmem_O_addr_0 =
            (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
             store_lane_gmem_O_d);
        int store_gmem_O_addr_1 =
            (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
             store_lane_gmem_O_d);
        LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(t_uptr_Z_0[0]);
        LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(t_uptr_Z_1[0]);
      }
    }
  }
}

template <const int kHeadDim, const int kStage>
void launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  // 根据 head_dim 选择 block tile 形状：
  //   d < 128  时，使用  64x64
  //   d >= 128 时，使用 128x128
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = (kHeadDim < 128) ? 4 : 8;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = (kHeadDim < 128) ? 4 : 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim < 128) ? 8 : 16;
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV =
      (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)); // 以 8 列为一组后的 head_dim 子块数
  constexpr int Br =
      kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // block 的行方向大小
  constexpr int Bc =
      kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; // block 的列方向大小
  constexpr int kNumThreads =
      WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // block 内线程总数
  constexpr int kPadQ = 0;
  constexpr int kPadK = 0;
  constexpr int kPadV = 0;
  // MMA 累加始终使用 fp32；这里只决定 R_D 是否继续以 fp32 暂存。
  // d 较小时保留 fp32，d 很大时改用 half 以控制寄存器压力。
  constexpr int kOStorageAccFloat32 = (kHeadDim < 256) ? 1 : 0;

  // Q/K 分别有 kStage 份缓冲区；单位仍是 half 元素个数。
  constexpr int QK_smem_size = (kStage * (Br * (kMmaAtomK + kPadQ)) +
                                kStage * (Bc * (kMmaAtomK + kPadK)));
  // V 在 P@V 阶段单独使用一片 shared memory。
  constexpr int V_smem_size = (kStage * (Bc * (kMmaAtomN * 2 + kPadV)));
  // QK^T 与 P@V 不同时执行，因此两阶段可以复用同一块 shared memory。
  const int smem_max_size = max(QK_smem_size, V_smem_size) * sizeof(half);

  const int QKV_batch = Q.size(0);
  const int QKV_head = Q.size(1);
  const int QKV_seqlen = Q.size(2);
  assert(QKV_seqlen % max(Br, Bc) == 0); // 当前实现要求 seqlen 与 tile 对齐

  // TODO: 可以继续尝试 block swizzle，进一步提升 L2 缓存命中率。
  // 当前 grid 把 Q tile 编号放在 x 维，batch*head 打平放在 y 维。
  dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head);
  dim3 block(kNumThreads); // 每个 block 对应 4 或 8 个 warp

  cudaFuncSetAttribute(
      flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
          kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      // 这里直接按实验环境中可用的动态 shared memory 上限设置。
      98304);

  flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_kernel<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
      kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
      kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>
      <<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
                                       reinterpret_cast<half *>(K.data_ptr()),
                                       reinterpret_cast<half *>(V.data_ptr()),
                                       reinterpret_cast<half *>(O.data_ptr()),
                                       QKV_seqlen, QKV_head);
}

void flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    int stages) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf) // Q: [B, H, N, D]
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf) // K: [B, H, N, D]
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf) // V: [B, H, N, D]
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf) // O: [B, H, N, D]
  const int d = Q.size(3);                  // head_dim

  // 目前支持 stage=1 或 stage=2。
  // stage>1 统一走双缓冲路径；否则走单缓冲路径。
  if (stages > 1) {
    switch (d) {
    case 32:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<32,
                                                                          2>(
          Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<64,
                                                                          2>(
          Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<96,
                                                                          2>(
          Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<128,
                                                                          2>(
          Q, K, V, O);
      break;
    case 256:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<256,
                                                                          2>(
          Q, K, V, O);
      break;
    case 512:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<512,
                                                                          2>(
          Q, K, V, O);
      break;
    case 1024:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<1024,
                                                                          2>(
          Q, K, V, O);
      break;
    default:
      throw std::runtime_error("暂不支持该 head_dim");
      break;
    }
  } else {
    switch (d) {
    case 32:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<32,
                                                                          1>(
          Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<64,
                                                                          1>(
          Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<96,
                                                                          1>(
          Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<128,
                                                                          1>(
          Q, K, V, O);
      break;
    case 256:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<256,
                                                                          1>(
          Q, K, V, O);
      break;
    case 512:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<512,
                                                                          1>(
          Q, K, V, O);
      break;
    case 1024:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv<1024,
                                                                          1>(
          Q, K, V, O);
      break;
    default:
      throw std::runtime_error("暂不支持该 head_dim");
      break;
    }
  }
}
