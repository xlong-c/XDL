#include "cuda_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <vector>
#include <vector_types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CHECK_CUDA(call)                                           \
  do {                                                             \
    cudaError_t err__ = (call);                                    \
    if (err__ != cudaSuccess) {                                    \
      std::fprintf(stderr, "CUDA error: %s @ %s:%d\n",             \
                   cudaGetErrorString(err__), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                     \
    }                                                              \
  } while (0)

#define CHECK_CUBLAS(call)                                          \
  do {                                                              \
    cublasStatus_t status__ = (call);                               \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                        \
      std::fprintf(stderr, "cuBLAS error: %d @ %s:%d\n",            \
                   static_cast<int>(status__), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                      \
    }                                                               \
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
  int load_a_smem_m = tid / 2;        // A 的行索引 m（两线程协作同一行）
  int load_a_smem_k = (tid & 1) << 2; // A 的列起点 k，取值 0 或 4（对应 float4）
  // B tile 布局: s_b[BK][BN]，每个线程负责 B 的 1 个 float4。
  int load_b_smem_k = tid / 32;        // B 的行索引 k（一个 warp 覆盖一行的 32 个 float4）
  int load_b_smem_n = (tid & 31) << 2; // B 的列起点 n，0,4,8,...,124

  // 当前 block 在全局矩阵中的基址偏移。
  int load_a_gmem_m = by * BM + load_a_smem_m; // A 的全局行号
  int load_b_gmem_n = bx * BN + load_b_smem_n; // B 的全局列起点

  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  // 沿 K 维分块：加载 A/B 子块 -> 线程内 FMA 累加到 r_c。
  for (int bk = 0; bk < kNumKTiles; ++bk) {
    // bk: 当前 K 维切片编号，对应区间 [bk*BK, bk*BK+BK)。
    int load_a_gmem_k = bk * BK + load_a_smem_k;              // A 在当前切片内的列起点
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k; // A[m, k:k+4]
    int load_b_gmem_k = bk * BK + load_b_smem_k;              // B 在当前切片内的行号
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
      const int store_c_gmem_m = by * BM + row_block * (BM / 2) + ty * kHalfTM + i;
      const int store_c_gmem_n = bx * BN + tx * kHalfTN;
      const int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
      const int r_row = row_block * kHalfTM + i;
      FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[r_row][0]);
      FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[r_row][kHalfTN]);
    }
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 1>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(
    float *A, float *B, float *C, const int M, const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  __shared__ float s_a[2][BK][BM + OFFSET]; // 双缓存
  __shared__ float s_b[2][BK][BN + OFFSET];

  float r_load_a[4];         // 寄存器
  float r_load_b[4];         // 寄存器, 放B在线程的当前列索引对应数据
  float r_comp_a[TM];        // 寄存器, 放A的行数据
  float r_comp_b[TN];        // 寄存器, 放B的列数据
  float r_c[TM][TN] = {0.0}; // 寄存器, 放结果C, M 行 N 列

  int load_a_smem_m = tid / 2;         // A 在 shared memory 的行索引 m（每 2 个线程对应 1 行）
  int load_a_smem_k = (tid & 1) << 2;  // A 在 shared memory 的列索引 k（每 2 个线程对应 1 列）
  int load_b_smem_n = (tid & 31) << 2; // B 在 shared memory 的列索引 n（按 warp
                                       // 内 lane 映射，向量宽度为 4）
  int load_b_smem_k = tid / 32;        // B 在 shared memory 的行索引 k（每个 warp 负责一行 BK 切片）

  int load_a_gmem_m = by * BM + load_a_smem_m; // A 在 global memory 的行坐标 m（块内行 + block 偏移）
  int load_a_gmem_n = bx * BN + load_b_smem_n; // 当前线程负责的 global 列坐标基址（用于 B 的 n 方向）
  {
    int load_a_gmem_k = load_a_smem_k;                        // A 的 k 起点，与 shared memory 的 k 布局保持一致
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k; // A 的线性地址 = m * K + k
    FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_addr]);        // 向量化读取 A 的连续 4 个 float 到寄存器
    int load_b_gmem_k = load_b_smem_k;                        // B 的 k 坐标由 warp 号决定
    int load_b_gmem_addr = load_b_gmem_k * N + load_a_gmem_n; // B 的线性地址 = k * N + n
    FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_addr]);        // 向量化读取 B 的连续 4 个 float 到寄存器

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0]; // 将 A 的第 0 个元素写入双缓冲 0
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1]; // 将 A 的第 1 个元素写入双缓冲 0
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2]; // 将 A 的第 2 个元素写入双缓冲 0
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3]; // 将 A 的第 3 个元素写入双缓冲 0

    FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b); // 将 B 的 4 元向量写入双缓冲 0
  }
  __syncthreads(); // 等待所有线程完成首块 BK 数据装载
  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int smem_sel = (bk - 1) & 1;                              // 当前用于计算的 shared memory 缓冲区编号（ping-pong）
    int smem_sel_next = bk & 1;                               // 当前迭代要预取写入的下一缓冲区编号
    int load_a_gmem_k = bk * BK + load_a_smem_k;              // 计算下一块 A 的 global k 起点
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k; // 下一块 A 的线性读取地址
    FLOAT4(r_load_a[0]) = FLOAT4(A[load_a_gmem_addr]);        // 预取下一块 A 到寄存器
    FLOAT4(r_load_b[0]) = FLOAT4(B[load_a_gmem_addr]);        // 预取下一块 B 到寄存器（当前实现复用同一地址变量）

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);          // 读取 A 子块上半行向量到计算寄存器
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + TM / 2]); // 读取 A 子块下半行向量到计算寄存器
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][ty * TN / 2]);          // 读取 B 子块左半列向量到计算寄存器
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][ty * TN / 2 + TN / 2]); // 读取 B 子块右半列向量到计算寄存器
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]); // 执行 FMA 累加：C += A*B
        }
      }
    }
    s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];             // 将预取 A 写入下一缓冲区（元素 0）
    s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];             // 将预取 A 写入下一缓冲区（元素 1）
    s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];             // 将预取 A 写入下一缓冲区（元素 2）
    s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];             // 将预取 A 写入下一缓冲区（元素 3）
    FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]); // 将预取 B 写入下一缓冲区
    __syncthreads();                                                                // 同步后再进入下一轮 bk，确保双缓冲数据可见
  }
// 计算剩下最后一块BK
#pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

struct KernelBenchmarkResult {
  const char *name;
  float max_abs_err;
  float avg_time_ms;
  double tflops;
  bool passed;
  cudaError_t cuda_status;
};

template <typename LaunchFunc>
KernelBenchmarkResult
benchmark_kernel(const char *name, LaunchFunc launch, float *d_c,
                 std::vector<float> &h_c, const std::vector<float> &h_ref,
                 const size_t bytes_c, const double total_flops,
                 const int warmup_iters, const int repeat_iters,
                 const float tolerance) {
  KernelBenchmarkResult result{name, 0.0f, 0.0f, 0.0, false, cudaSuccess};
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int iter = 0; iter < warmup_iters; ++iter) {
    CHECK_CUDA(cudaMemset(d_c, 0, bytes_c));
    launch();
    result.cuda_status = cudaGetLastError();
    if (result.cuda_status != cudaSuccess) {
      goto cleanup;
    }
    result.cuda_status = cudaDeviceSynchronize();
    if (result.cuda_status != cudaSuccess) {
      goto cleanup;
    }
  }

  for (int iter = 0; iter < repeat_iters; ++iter) {
    CHECK_CUDA(cudaMemset(d_c, 0, bytes_c));
    CHECK_CUDA(cudaEventRecord(start));
    launch();
    result.cuda_status = cudaGetLastError();
    if (result.cuda_status != cudaSuccess) {
      goto cleanup;
    }
    CHECK_CUDA(cudaEventRecord(stop));
    result.cuda_status = cudaEventSynchronize(stop);
    if (result.cuda_status != cudaSuccess) {
      goto cleanup;
    }

    float iter_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
    result.avg_time_ms += iter_ms;
  }

  result.avg_time_ms /= static_cast<float>(repeat_iters);
  result.tflops = total_flops / (static_cast<double>(result.avg_time_ms) * 1.0e9);

  CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < h_c.size(); ++i) {
    result.max_abs_err = std::max(result.max_abs_err, std::fabs(h_c[i] - h_ref[i]));
  }
  result.passed = result.max_abs_err <= tolerance;

cleanup:
  if (result.cuda_status != cudaSuccess) {
    cudaGetLastError();
  }
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return result;
}

int main() {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 512;
  constexpr float kTolerance = 1e-3f;
  constexpr int kWarmupIters = 3;
  constexpr int kRepeatIters = 10;

  const size_t size_a = static_cast<size_t>(M) * K;
  const size_t size_b = static_cast<size_t>(K) * N;
  const size_t size_c = static_cast<size_t>(M) * N;
  const size_t bytes_a = size_a * sizeof(float);
  const size_t bytes_b = size_b * sizeof(float);
  const size_t bytes_c = size_c * sizeof(float);
  const double total_flops = 2.0 * static_cast<double>(M) *
                             static_cast<double>(N) * static_cast<double>(K);

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

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += h_a[m * K + k] * h_b[k * N + n];
      }
      h_ref[m * N + n] = sum;
    }
  }

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, bytes_a));
  CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
  CHECK_CUDA(cudaMalloc(&d_c, bytes_c));

  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice));

  const dim3 block_naive(32, 32);
  const dim3 grid_naive((N + 32 - 1) / 32, (M + 32 - 1) / 32);
  const dim3 block_tiled(16, 16);
  const dim3 grid_tiled((N + 128 - 1) / 128, (M + 128 - 1) / 128);

  std::vector<KernelBenchmarkResult> results;
  results.push_back(benchmark_kernel(
      "sgemm_naive_f32",
      [&]() {
        sgemm_naive_f32_kernel<<<grid_naive, block_naive>>>(d_a, d_b, d_c, M, N,
                                                            K);
      },
      d_c, h_c, h_ref, bytes_c, total_flops, kWarmupIters, kRepeatIters,
      kTolerance));
  results.push_back(benchmark_kernel(
      "sgemm_sliced_k_f32",
      [&]() {
        sgemm_sliced_k_f32_kernel<<<grid_naive, block_naive>>>(d_a, d_b, d_c, M,
                                                               N, K);
      },
      d_c, h_c, h_ref, bytes_c, total_flops, kWarmupIters, kRepeatIters,
      kTolerance));
  results.push_back(benchmark_kernel(
      "sgemm_t_8x8_f32x4",
      [&]() {
        sgemm_t_8x8_sliced_k_f32x4_kernel<<<grid_tiled, block_tiled>>>(
            d_a, d_b, d_c, M, N, K);
      },
      d_c, h_c, h_ref, bytes_c, total_flops, kWarmupIters, kRepeatIters,
      kTolerance));
  results.push_back(benchmark_kernel(
      "sgemm_t_8x8_f32x4_bcf<O=4>",
      [&]() {
        sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<128, 128, 8, 8, 8, 4>
            <<<grid_tiled, block_tiled>>>(d_a, d_b, d_c, M, N, K);
      },
      d_c, h_c, h_ref, bytes_c, total_flops, kWarmupIters, kRepeatIters,
      kTolerance));
  results.push_back(benchmark_kernel(
      "sgemm_t_8x8_f32x4_dbuf<O=4>",
      [&]() {
        sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<128, 128, 8, 8, 8, 4>
            <<<grid_tiled, block_tiled>>>(d_a, d_b, d_c, M, N, K);
      },
      d_c, h_c, h_ref, bytes_c, total_flops, kWarmupIters, kRepeatIters,
      kTolerance));

  std::printf("SGEMM benchmark (M=%d, N=%d, K=%d, FLOPs=%.0f)\n", M, N, K,
              total_flops);
  std::printf("%-30s | %12s | %12s | %12s | %12s\n", "Kernel", "MaxAbsErr",
              "Time(ms)", "TFLOPS", "Status");
  std::printf("-------------------------------+--------------+--------------+--"
              "------------+--------------\n");

  bool all_passed = true;
  for (const auto &result : results) {
    if (result.cuda_status == cudaSuccess) {
      std::printf("%-30s | %12.6g | %12.4f | %12.4f | %12s\n", result.name,
                  result.max_abs_err, result.avg_time_ms, result.tflops,
                  result.passed ? "PASS" : "FAIL");
      all_passed = all_passed && result.passed;
      continue;
    }

    std::printf("%-30s | %12s | %12s | %12s | %12s\n", result.name, "N/A",
                "N/A", "N/A", "CUDA_ERR");
    std::printf("  -> CUDA error: %s\n",
                cudaGetErrorString(result.cuda_status));
    all_passed = false;
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
