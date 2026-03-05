#include "cuda_runtime.h"
#if defined(__clang__)
#include <__clang_cuda_runtime_wrapper.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <vector_types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error: %s @ %s:%d\n",                         \
                   cudaGetErrorString(err__), __FILE__, __LINE__);             \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
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
  // 每行需要 128/4=32 个线程(每线程 1 个 float4)完成, 因此 tid/32 映射到 B 的行
  // k(0~7).
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
          // 该版本可能出现 shared memory bank conflict:
          // - s_a 布局是 [BM][BK]=[128][8]，行跨度 stride=BK=8(float)
          // - warp 内通常有两组 ty(例如 ty=0/1)，会同时访问两行:
          //   row0=0*TM+m, row1=1*TM+m=row0+8
          // - bank 号可近似看作 bank=(row*stride+k)%32
          //   两行差值: (row1-row0)*stride = 8*8 = 64, 64%32=0
          //   => 两个不同地址落到同一 bank，形成 2-way conflict
          // 这会降低 shared memory 吞吐，尤其在 k 循环高频读取时更明显。
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
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(C[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void
sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float *A, float *B, float *C, const int M,
                                      const int N, const int K) {
  constexpr int kHalfTM = TM / 2;
  constexpr int kHalfTN = TN / 2;
  const int kNumKTiles = (K + BK - 1) / BK;

  // 每个 block 负责 C 的一个 BM x BN tile；每个线程计算 TM x TN 的寄存器子块。
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // block 内线性线程号：[0, blockDim.x * blockDim.y)
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[BK][BM + OFFSET];
  __shared__ float s_b[BK][BN + OFFSET];

  // 向量化加载缓冲：一次搬运 4 个 float。
  float r_load_a[4];
  float r_load_b[4];
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};

  // 线程到共享内存坐标的映射（按 float4 对齐）。
  // A tile 布局: s_a[BK][BM]，这里每个线程负责 A 的 4 个连续 k 元素。
  int load_a_smem_m = tid / 2; // A 的行索引 m（两线程协作同一行）
  int load_a_smem_k = (tid & 1)
                      << 2; // A 的列起点 k，取值 0 或 4（对应 float4）
  // B tile 布局: s_b[BK][BN]，每个线程负责 B 的 1 个 float4。
  int load_b_smem_k =
      tid / 32; // B 的行索引 k（一个 warp 覆盖一行的 32 个 float4）
  int load_b_smem_n = (tid & 31) << 2; // B 的列起点 n，0,4,8,...,124

  // 当前 block 在全局矩阵中的基址偏移。
  int load_a_gmem_m = by * BM + load_a_smem_m; // A 的全局行号
  int load_b_gmem_n = bx * BN + load_b_smem_n; // B 的全局列起点

  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  // 沿 K 维分块：加载 A/B 子块 -> 线程内 FMA 累加到 r_c。
  for (int bk = 0; bk < kNumKTiles; ++bk) {
    // bk: 当前 K 维切片编号，对应区间 [bk*BK, bk*BK+BK)。
    int load_a_gmem_k = bk * BK + load_a_smem_k; // A 在当前切片内的列起点
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k; // A[m, k:k+4]
    int load_b_gmem_k = bk * BK + load_b_smem_k; // B 在当前切片内的行号
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n; // B[k, n:n+4]
    FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_addr]);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      s_a[load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
    FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; ++tk) {
      // tk: 切片内的 K 偏移。每次取 A/B 一条 k 维向量做外积累加。
      const int comp_a_base = ty * kHalfTM;
      const int comp_b_base = tx * kHalfTN;
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][comp_a_base]);
      FLOAT4(r_comp_a[kHalfTM]) = FLOAT4(s_a[tk][comp_a_base + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][comp_b_base]);
      FLOAT4(r_comp_b[kHalfTN]) = FLOAT4(s_b[tk][comp_b_base + BN / 2]);

#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          // tm/tn: 线程内寄存器子块 r_c[TM][TN] 的局部行列索引。
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int row_block = 0; row_block < 2; ++row_block) {
#pragma unroll
    for (int i = 0; i < kHalfTM; ++i) {
      const int store_c_gmem_m =
          by * BM + row_block * (BM / 2) + ty * kHalfTM + i;
      const int store_c_gmem_n = bx * BN + tx * kHalfTN;
      const int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
      const int r_row = row_block * kHalfTM + i;
      FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[r_row][0]);
      FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[r_row][kHalfTN]);
    }
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(
    float *A, float *B, float *C, const int M, const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  __shared__ float s_a[2][BK][BM + OFFSET]; // 双缓存
  __shared__ float s_b[2][BK][BN + OFFSET];

  float r_load_a[4];  // 寄存器
  float r_load_b[4];  // 寄存器, 放B在线程的当前列索引对应数据
  float r_comp_a[TM]; // 寄存器, 放A的行数据
  float r_comp_b[TN]; // 寄存器, 放B的列数据
  float r_c[TM][TN] = {0.0}; // 寄存器, 放结果C, M 行 N 列

  int load_a_smem_m =
      tid / 2; // A 在 shared memory 的行索引 m（每 2 个线程对应 1 行）
  int load_a_smem_k =
      (tid & 1) << 2; // A 在 shared memory 的列索引 k（每 2 个线程对应 1 列）
  int load_b_smem_n = (tid & 31) << 2;
  int load_b_smem_k = tid / 32;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_a_gmem_n = bx * BN + load_b_smem_n;

  int load_a_gmem_k = load_a_smem_k;
  int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
  FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_addr]);
  int load_b_gmem_k = load_b_smem_k;
  int load_b_gmem_addr = load_b_gmem_k * N + load_a_gmem_n;
  FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_addr]);

  s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
  s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
  s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
  s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

  FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b);

  __syncthreads();

  

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
    std::fprintf(stderr, "No CUDA device found.\n");
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

  std::printf("max_abs_err = %.8g\n", max_abs_err);
  if (max_abs_err > 1e-3f) {
    std::fprintf(stderr, "Validation failed.\n");
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  std::printf("Validation passed.\n");
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return EXIT_SUCCESS;
}
