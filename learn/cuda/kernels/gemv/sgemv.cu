#include "algorithm"

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <vector_types.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])

template <const int KWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  float res = val;
  for (int offset = KWarpSize >> 1; offset >= 1; offset >>= 2) {
    res += __shfl_down_sync(0xffffffff, res, offset);
  }
  return res;
}

// A: MxK B: Kx1 C: Mx1
__global__ void sgemv_k32_f32_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;

  if (m >= M)
    return;
  float sum = 0.0f;
  int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
  for (int w = 0; w < NUM_WARPS; w += 32) {
    int k = m * WARP_SIZE + lane;
    sum += A[m * K + k] * B[k];
  }
  sum = warp_reduce_sum_f32(sum);
  if (lane == 0)
    C[m] = sum;
}

__global__ void sgmv_k128_f32x4_kernel(float *a, float *x, float *y, int M,
                                       int K) {
  int tx = threadIdx.x; // 0~31
  int ty = threadIdx.y; // 0~3
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;

  if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; w += 4) {
      int k = m * WARP_SIZE + lane;
      float4 reg_x = FLOAT4(x[k]);
      float4 reg_a = FLOAT4(a[m * K + k]);
      sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z +
              reg_a.w * reg_x.w);
    }
    sum = warp_reduce_sum_f32(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

/// @brief 小 K 值优化的 SGEMV kernel (K <= 32, K ~ 16 时最优)
/// @tparam ROW_PER_WARP 每个 warp 处理的行数 (默认：2)
/// @param a 输入矩阵 A (M x K)，行优先存储
/// @param x 输入向量 B (K x 1)
/// @param y 输出向量 C (M x 1)
/// @param M 矩阵 A 的行数
/// @param N 未使用 (API 兼容保留)
/// @param K 矩阵 A 的列数 (必须 <= 32, 16 左右最优)
///
/// @note 线程映射逻辑:
///   - 每个 warp 处理 ROW_PER_WARP 行 (默认：2 行)
///   - 每行需要 K_WARP_SIZE = WARP_SIZE / ROW_PER_WARP 个线程 (默认：16)
///   - lane 0-15 -> 第 m 行，lane 16-31 -> 第 m+1 行
///   - 每个线程计算 a[m,k] * x[k]，其中 k = lane % K_WARP_SIZE
///   - warp_reduce 在行组内对所有 k 值求和
///   - 仅 k == 0 的线程写入最终结果
///
/// @note K 必须 <= K_WARP_SIZE (默认 16)。若 K < K_WARP_SIZE，
///   仅计算前 K 个元素，其余补零
template <const int ROW_PER_WARP = 2>
__global__ void sgemv_k16_f32_kernel(float *a, float *x, float *y, int M, int N,
                                     int K) {
  // K_WARP_SIZE: 每个 warp 内处理单行的线程数
  // = 32 / 2 = 16 (当 ROW_PER_WARP=2)
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  
  // k: 列索引，范围 [0, K_WARP_SIZE)
  int k = lane % K_WARP_SIZE;
  
  // m: 行索引
  // - blockDim.y * bx + ty: block 级别的行偏移
  // - lane / K_WARP_SIZE: warp 内行选择 (ROW_PER_WARP=2 时为 0 或 1)
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;

  if (m < M) {
    // 边界检查：确保 k 在有效范围 [0, K) 内
    if (k < K) {
      float sum = a[m * K + k] * x[k];
      sum = warp_reduce_sum_f32<K_WARP_SIZE>(sum);
      // 仅 k==0 的线程写入结果 (行组内所有线程的 sum 相同)
      if (k == 0)
        y[m] = sum;
    }
  }
}
