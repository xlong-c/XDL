#include "algorithm"
#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <__clang_cuda_runtime_wrapper.h>
#include <vector_types.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void sgemm_naive_f32_kernel(float *A, float *B, float *C, int M,
                                       int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
#pragma unroll 8
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

/**
 * @brief SGEMM kernel with K-slicing and shared memory tiling
 *
 * 实现矩阵乘法 C = A * B，其中：
 *   - A: M x K 矩阵
 *   - B: K x N 矩阵
 *   - C: M x N 矩阵
 *
 * 优化策略：
 *   [1] 2D Block Tiling: 每个 thread block 计算 C 矩阵的 BM x BN 子块
 *   [2] K-Slicing: 将 K 维度按 BK 大小分片，逐步累加乘积
 *   [3] Shared Memory: 每个迭代加载 A 的 BM x BK 和 B 的 BK x BN 子块
 *       到共享内存，减少全局内存访问次数
 *   [4] Coalescing: warp 内线程连续读取全局内存，最大化带宽利用
 *
 * @tparam BM Block size along M dimension (default: 32)
 * @tparam BN Block size along N dimension (default: 32)
 * @tparam BK Block size along K dimension (default: 32)
 *
 * @param A 指针 to M x K 矩阵 (row-major)
 * @param B 指针 to K x N 矩阵 (row-major)
 * @param C 指针 to M x N 输出矩阵 (row-major)
 * @param M 矩阵 A 和 C 的行数
 * @param N 矩阵 B 和 C 的列数
 * @param K 矩阵 A 的列数和 B 的行数
 */
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *A, float *B, float *C, int M,
                                          int N, int K) {
  __shared__ float s_a[BM][BK], s_b[BK][BN];

  // 计算 block 和 thread 索引
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;
  // 计算共享内存加载索引
  int load_smem_a_m = tid / 32;
  int load_smem_a_k = tid % 32;
  int load_smem_b_k = tid / 32;
  int load_smem_b_n = tid % 32;
  // 计算全局内存加载索引
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  float sum = 0.f;
  // K-slicing 循环：分片累加
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载 A 矩阵子块到共享内存
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = A[load_gmem_a_addr];
    // 加载 B 矩阵子块到共享内存
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = B[load_gmem_b_addr];
    __syncthreads();
    // 矩阵乘法计算：C += A * B
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads();
  }
  // 将结果写回全局内存
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  C[store_gmem_c_addr] = sum;
}
