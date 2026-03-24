#include "algorithm"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "cuda_runtime.h"
#include <__clang_cuda_runtime_wrapper.h>
#include <ctime>
#include <curand_mtgp32_kernel.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// 这是一个简单的半精度矩阵乘法核函数,每个线程计算C矩阵中的一个元素
// 每个线程需要读取Kx2个半精度数值,并进行K次乘加操作,最后将结果写回C矩阵
// 算术强度AI = 2K / (4K + 2) = K / (2K + 1)  = 0.5 FLOP/Byte
__global__ void hgemm_native(const half *A, const half *B, half *C, int M, int N, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; //全局x坐标,对应C矩阵的列
  int idy = blockIdx.y * blockDim.y + threadIdx.y; //全局y坐标,对应C矩阵的行
  if (idx < N && idy < M) {
    half sum = CUDART_ZERO_FP16; //累加器,使用fp16
    for (int k = 0; k < K; k++) {
      sum += A[idy * K + k] * B[k * N + idx];
    }
    C[idy * N + idx] = sum;
  }
}

// shared memory 分块版本:
// - 每个 block 负责 C 的一个 TILE_M x TILE_N 子块
// - 沿 K 维按 TILE_K 切片, 协作搬运 A/B 子块到 shared memory 后复用
// - 全局访存的算术强度:
//   AI = TILE_M * TILE_N * K / (K * (TILE_M + TILE_N) + TILE_M * TILE_N)
//   默认 TILE_M=TILE_N=32 时, AI = 16K / (K + 16), K 足够大时趋近 16 FLOP/Byte
// 建议 launch:
//   dim3 block(TILE_N, TILE_M);
//   dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
template <const int TILE_M = 32, const int TILE_N = 32, const int TILE_K = 8>
__global__ void hgemm_sharemem(const half *A, const half *B, half *C, int M, int N, int K) {
  __shared__ half s_a[TILE_M][TILE_K];
  __shared__ half s_b[TILE_K][TILE_N];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx; //线程在块内的线性id
  int num_threads = blockDim.x * blockDim.y;

  int row = by * TILE_M + ty;
  int col = bx * TILE_N + tx;
  half sum = CUDART_ZERO_FP16; //使用 fp16 累加

  for (int tile_k_base = 0; tile_k_base < K; tile_k_base += TILE_K) {
    // 协作加载 A 的 TILE_M x TILE_K 子块
    for (int load_idx = tid; load_idx < TILE_M * TILE_K; load_idx += num_threads) {
      int smem_m = load_idx / TILE_K;
      int smem_k = load_idx % TILE_K;
      int gmem_m = by * TILE_M + smem_m;
      int gmem_k = tile_k_base + smem_k;
      s_a[smem_m][smem_k] =
          (gmem_m < M && gmem_k < K) ? A[gmem_m * K + gmem_k] : CUDART_ZERO_FP16;
    }

    // 协作加载 B 的 TILE_K x TILE_N 子块
    for (int load_idx = tid; load_idx < TILE_K * TILE_N; load_idx += num_threads) {
      int smem_k = load_idx / TILE_N;
      int smem_n = load_idx % TILE_N;
      int gmem_k = tile_k_base + smem_k;
      int gmem_n = bx * TILE_N + smem_n;
      s_b[smem_k][smem_n] =
          (gmem_k < K && gmem_n < N) ? B[gmem_k * N + gmem_n] : CUDART_ZERO_FP16;
    }
    __syncthreads();

    if (ty < TILE_M && tx < TILE_N && row < M && col < N) {
#pragma unroll
      for (int k = 0; k < TILE_K; ++k) {
        sum = __hadd(sum, __hmul(s_a[ty][k], s_b[k][tx]));
      }
    }
    __syncthreads();
  }

  if (ty < TILE_M && tx < TILE_N && row < M && col < N) {
    C[row * N + col] = sum;
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_shared_f16x4(
    half *A,
    half *B,
    half *C,
    const int M,
    const int N,
    const int K) {
  // 静态断言确保参数合法
  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert(BN % TN == 0, "BN must be divisible by TN");
  static_assert(BK % 4 == 0, "BK must be divisible by 4 for float4 loading");

  // 线程块大小: 256 个线程, 配置为 16x16
  // 每个 block 处理 BM x BN = 128 x 128 的 C 子块
  // 每个线程计算 TM x TN = 8 x 8 个输出元素
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx; // tid in [0, 255]

  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Shared memory for A and B tiles
  __shared__ half s_a[BM][BK]; // 128 x 8 = 1024 half
  __shared__ half s_b[BK][BN]; // 8 x 128 = 1024 half

  // ========== 全局内存索引计算 ==========
  // A 矩阵: 加载 BM x BK = 128 x 8 = 1024 half
  // 使用 float4 向量化加载, 需要 1024 / 8 = 128 次加载
  // 每个线程加载 128 / 256 = 0.5 次, 即 2 个线程协作加载 1 个 float4
  int load_a_start_m = by * BM; // 第m行
  int load_a_k = 0;

  // B 矩阵: 加载 BK x BN = 8 x 128 = 1024 half
  // 使用 float4 向量化加载, 需要 1024 / 8 = 128 次加载
  // 每个线程加载 128 / 256 = 0.5 次, 即 2 个线程协作加载 1 个 float4
  int load_b_start_n = bx * BN; //第n列
  int load_b_k = 0;

  // 计算当前线程负责的 C 矩阵输出位置
  int thread_m = by * BM + ty * TM; // 线程输出的起始行
  int thread_n = bx * BN + tx * TN; // 线程输出的起始列

  // 寄存器存储累加结果 (TM x TN = 8 x 8)
  half frag_a[TM];                         // A 矩阵片段
  half frag_b[TN];                         // B 矩阵片段
  half accum[TM][TN] = {CUDART_ZERO_FP16}; // 累加器

  // 计算当前线程相对于块内的地址,s_a是BMxBK:128x8
  // 每行8个数据,每个线程计算4个数据,每行两个
  int idx_s_am = tid / (BK / 4);
  int idx_s_ak = tid % (BK / 4) * 4;
  // s_b是BKxBN:8x128
  // 每行128个数据,每个线程计算4个,一行需要32个线程
  int idx_s_bk = tid / (BN / 4);
  int idx_s_bn = tid % (BN / 4) * 4;

  // 计算当前块的对应的行和列
  int idx_g_am = by * BM + idx_s_am;
  int idx_g_bn = bx * BN + idx_s_bn;
  const half2 zero2 = __halves2half2(CUDART_ZERO_FP16, CUDART_ZERO_FP16);

  for (int bk = 0; bk < K; bk += BK) {
    int idx_g_ak = bk + idx_s_ak;
    int idx_g_bk = bk + idx_s_bk; // 计算 B 的全局 k 维索引

    // 向量化加载 A: 每次写入 2 个 half
    if (idx_g_am < M && idx_g_ak < K) {
      if (idx_g_ak + 1 < K) {
        HALF2(s_a[idx_s_am][idx_s_ak + 0]) = HALF2(A[idx_g_am * K + idx_g_ak + 0]);
      } else {
        s_a[idx_s_am][idx_s_ak + 0] = A[idx_g_am * K + idx_g_ak + 0];
        s_a[idx_s_am][idx_s_ak + 1] = CUDART_ZERO_FP16;
      }
    } else {
      HALF2(s_a[idx_s_am][idx_s_ak + 0]) = zero2;
    }
    if (idx_g_am < M && idx_g_ak + 2 < K) {
      if (idx_g_ak + 3 < K) {
        HALF2(s_a[idx_s_am][idx_s_ak + 2]) = HALF2(A[idx_g_am * K + idx_g_ak + 2]);
      } else {
        s_a[idx_s_am][idx_s_ak + 2] = A[idx_g_am * K + idx_g_ak + 2];
        s_a[idx_s_am][idx_s_ak + 3] = CUDART_ZERO_FP16;
      }
    } else {
      HALF2(s_a[idx_s_am][idx_s_ak + 2]) = zero2;
    }

    // 向量化加载 B: B 的线性索引是 [k, n] => k * N + n
    if (idx_g_bk < K && idx_g_bn < N) {
      if (idx_g_bn + 1 < N) {
        HALF2(s_b[idx_s_bk][idx_s_bn + 0]) = HALF2(B[idx_g_bk * N + idx_g_bn + 0]);
      } else {
        s_b[idx_s_bk][idx_s_bn + 0] = B[idx_g_bk * N + idx_g_bn + 0];
        s_b[idx_s_bk][idx_s_bn + 1] = CUDART_ZERO_FP16;
      }
    } else {
      HALF2(s_b[idx_s_bk][idx_s_bn + 0]) = zero2;
    }
    if (idx_g_bk < K && idx_g_bn + 2 < N) {
      if (idx_g_bn + 3 < N) {
        HALF2(s_b[idx_s_bk][idx_s_bn + 2]) = HALF2(B[idx_g_bk * N + idx_g_bn + 2]);
      } else {
        s_b[idx_s_bk][idx_s_bn + 2] = B[idx_g_bk * N + idx_g_bn + 2];
        s_b[idx_s_bk][idx_s_bn + 3] = CUDART_ZERO_FP16;
      }
    } else {
      HALF2(s_b[idx_s_bk][idx_s_bn + 2]) = zero2;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          int idx_temp_am = ty * TM + m;
          int idx_temp_bn = tx * TN + n;
          accum[m][n] += s_a[idx_temp_am][k] * s_b[k][idx_temp_bn];
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int idx_cm = by * BM + ty * TM + m;
    if (idx_cm >= M) {
      continue;
    }
#pragma unroll
    for (int n = 0; n < TN; n += 2) {
      int idx_cn = bx * BN + tx * TN + n;
      if (idx_cn + 1 < N) {
        int idx_c = idx_cm * N + idx_cn;
        HALF2(C[idx_c]) = HALF2(accum[m][n]);
      } else if (idx_cn < N) {
        C[idx_cm * N + idx_cn] = accum[m][n];
      }
    }
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_shared_f16x4_pack_bank(half *A, half *B, half *C, int M, int N, int K) {
  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert(BN % TN == 0, "BN must be divisible by TN");
  static_assert(BK % 4 == 0, "BK must be divisible by 4 for float4 loading");
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  __shared__ half s_a[BM][BK]; // 128 x 8 = 1024 half
  __shared__ half s_b[BK][BN]; // 8 x 128 = 1024 half

  int idx_smem_am = tid / (BK / 4);     // 计算当前线程在 s_a 中的行坐标
  int idx_smem_ak = tid % (BK / 4) * 4; // 列坐标
  int idx_smem_bk = tid / (BN / 4);
  int idx_smem_bn = tid % (BN / 4) * 4;

  int idx_g_am = by * BM + idx_smem_am; // 线程对应的全局行坐标, by是第y块, 一块y是BM行
  int idx_g_bn = bx * BN + idx_smem_bn;
  const half2 zero2 = __halves2half2(CUDART_ZERO_FP16, CUDART_ZERO_FP16);

  if (idx_g_am >= M && idx_g_bn >= N)
    return;

  half r_c[TM][TN] = {CUDART_ZERO_FP16};
  // 在K上进行完整的循环,直到完成所有的值
  for (int bk = 0; bk < K; bk += BK) {
    // 这里的k是循环的所以放到里面来算
    int idx_g_ak = bk + idx_smem_ak;
    int idx_g_bk = bk + idx_smem_bk;
    int addr_g_a = K * idx_g_am + idx_g_ak;
    int addr_g_b = N * idx_g_bk + idx_g_bn;

    LDST64BITS(s_a[idx_smem_am][idx_smem_ak]) = LDST64BITS(A[addr_g_a]);
    LDST64BITS(s_b[idx_smem_bk][idx_smem_bn]) = LDST64BITS(B[addr_g_b]);

    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          int idx_tam = ty * TM + m;
          int idx_tbn = tx * TN + n;
          r_c[m][n] = __hfma(s_a[idx_tam][k], s_b[k][idx_tbn]);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < TM; m++)
#pragma unroll
      for (int n = 0; n < TN; n++) {
        int idx_g_cn = bx * BN + tx * TN + n;
        int addr_g_c = idx_g_cn * N + idx_g_cn;
        LDST64BITS(C[addr_g_c]) = LDST64BITS(r_c[m][n]);
      }
  }
}
