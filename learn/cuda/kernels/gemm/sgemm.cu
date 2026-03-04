#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <vector_types.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " @ "       \
                << __FILE__ << ":" << __LINE__ << std::endl;                  \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

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

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *A, float *B, float *C,
                                                  int M, int N, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = blockDim.x * ty + tx;

  __shared__ float s_a[BM][BK]; // [128,8]
  __shared__ float s_b[BK][BN]; // [8,128]

  // A tile 为 [BM, BK] = [128, 8], 每行有 8 个 float.
  // 使用 float4 向量化加载时, 每次搬运 4 个 float, 因此一行需要 2 次加载.
  // 这里令相邻两个线程协作同一行: tid/2 给出行号 m(0~127), 共覆盖 BM 行.
  int load_smem_a_m = tid / 2;
  // tid 为偶数线程加载该行前半段 k=0~3, 奇数线程加载后半段 k=4~7.
  // 后续通过 FLOAT4(...) 一次写入 4 个连续元素.
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;
  // B tile 为 [BK, BN] = [8, 128], 每行有 128 个 float.
  // 每行需要 128/4=32 个线程(每线程 1 个 float4)完成, 因此 tid/32 映射到 B 的行 k(0~7).
  int load_smem_b_k = tid / 32;
  // warp 内 lane(=tid%32) 决定该线程在 B 行内的列起点.
  // 乘 4 是因为 float4 对齐: n 起点依次为 0,4,8,...,124.
  int load_smem_b_n = tid % 32 * 4;

  // 将 block 内局部行号映射到全局 A 的实际行号.
  // by*BM 是当前 block 在 M 维覆盖的起始行, load_smem_a_m 是块内偏移.
  int load_gmem_a_m = by * BM + load_smem_a_m;
  // 将 block 内局部列偏移映射到全局 B 的实际列起点(按 float4 对齐).
  // bx*BN 是当前 block 在 N 维覆盖的起始列.
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  // 每个线程持有一个 TMxTN 的寄存器累加块, 保存该线程负责的 C 子块部分和.
  // 初始化为 0, 在 K-slicing 循环中持续累加.
  float r_c[TM][TN] = {0.0f};

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {

    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(A[load_gmem_a_addr]);
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(B[load_gmem_b_addr]);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          int comp_smem_a_m = ty * TM + m;
          int comp_smem_b_n = tx * TN + n;
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < TM; m++) {
    int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int storegmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(C[storegmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

int main() {
  constexpr int M = 64;
  constexpr int N = 64;
  constexpr int K = 64;

  const size_t size_a = static_cast<size_t>(M) * K;
  const size_t size_b = static_cast<size_t>(K) * N;
  const size_t size_c = static_cast<size_t>(M) * N;

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::cerr << "No CUDA device found." << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<float> h_a(size_a);
  std::vector<float> h_b(size_b);
  std::vector<float> h_c(size_c, 0.0f);
  std::vector<float> h_ref(size_c, 0.0f);

  for (size_t i = 0; i < size_a; ++i) {
    h_a[i] = static_cast<float>((i % 13) - 6) * 0.1f;
  }
  for (size_t i = 0; i < size_b; ++i) {
    h_b[i] = static_cast<float>((i % 17) - 8) * 0.1f;
  }

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_c, 0, size_c * sizeof(float)));

  dim3 block(32, 32);
  dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32);
  sgemm_sliced_k_f32_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, size_c * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += h_a[m * K + k] * h_b[k * N + n];
      }
      h_ref[m * N + n] = sum;
    }
  }

  float max_abs_err = 0.0f;
  for (size_t i = 0; i < size_c; ++i) {
    max_abs_err = std::max(max_abs_err, std::fabs(h_c[i] - h_ref[i]));
  }

  std::cout << "max_abs_err = " << max_abs_err << std::endl;
  if (max_abs_err > 1e-3f) {
    std::cerr << "Validation failed." << std::endl;
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  std::cout << "Validation passed." << std::endl;
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return EXIT_SUCCESS;
}
