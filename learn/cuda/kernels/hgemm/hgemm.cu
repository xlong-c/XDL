#include "algorithm"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "cuda_runtime.h"
#include <__clang_cuda_runtime_wrapper.h>
#include <curand_mtgp32_kernel.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(value))
#define FLOAT4C(value) (reinterpret_cast<const float4 *>(value))
#define HALF2(value) (reinterpret_cast<half2 *>(value))
#define HALF2C(value) (reinterpret_cast<const half2 *>(value))
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(value))
#define WARP_SIZE 32

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
    const half *A,
    const half *B,
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
  half frag_a[TM];    // A 矩阵片段
  half frag_b[TN];    // B 矩阵片段
  half accum[TM][TN] = {}; // 累加器

  // ========== 主循环: 沿 K 维分块处理 ==========
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // ========== 协作加载 A 矩阵到 shared memory ==========
    // 256 个线程协作加载 128 x 8 = 1024 half
    // 使用 float4 (8 half) 向量化加载, 共需 128 次加载
    // 每 2 个线程负责 1 个 float4
#pragma unroll
    for (int i = 0; i < BM * BK / (256 * 8); ++i) {
      int load_idx = i * 256 + tid;           // 加载索引
      int smem_m = load_idx / (BK / 8);       // shared memory 行
      int smem_k = (load_idx % (BK / 8)) * 8; // shared memory 列 (8 half 对齐)

      int gmem_m = load_a_start_m + smem_m;
      int gmem_k = k_tile + smem_k;

      // 使用 float4 向量化加载 8 个 half
      if (gmem_m < M && gmem_k + 7 < K) {
        *FLOAT4(&s_a[smem_m][smem_k]) = *FLOAT4C(&A[gmem_m * K + gmem_k]);
      } else {
        // 边界处理: 逐个元素加载
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          s_a[smem_m][smem_k + j] =
              (gmem_m < M && gmem_k + j < K) ? A[gmem_m * K + gmem_k + j] : CUDART_ZERO_FP16;
        }
      }
    }

    // ========== 协作加载 B 矩阵到 shared memory ==========
    // 256 个线程协作加载 8 x 128 = 1024 half
    // 使用 float4 (8 half) 向量化加载, 共需 128 次加载
#pragma unroll
    for (int i = 0; i < BK * BN / (256 * 8); ++i) {
      int load_idx = i * 256 + tid;
      int smem_k = load_idx / (BN / 8);       // shared memory 行
      int smem_n = (load_idx % (BN / 8)) * 8; // shared memory 列 (8 half 对齐)

      int gmem_k = k_tile + smem_k;
      int gmem_n = load_b_start_n + smem_n;

      // 使用 float4 向量化加载 8 个 half
      if (gmem_k < K && gmem_n + 7 < N) {
        *FLOAT4(&s_b[smem_k][smem_n]) = *FLOAT4C(&B[gmem_k * N + gmem_n]);
      } else {
        // 边界处理
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          s_b[smem_k][smem_n + j] =
              (gmem_k < K && gmem_n + j < N) ? B[gmem_k * N + gmem_n + j] : CUDART_ZERO_FP16;
        }
      }
    }

    __syncthreads();

    // ========== 计算: 从 shared memory 加载数据并计算 ==========
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // 加载 A 矩阵片段到寄存器 (TM = 8 个 half)
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        frag_a[i] = s_a[ty * TM + i][k];
      }

      // 加载 B 矩阵片段到寄存器 (TN = 8 个 half)
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        frag_b[j] = s_b[k][tx * TN + j];
      }

      // 使用 half2 向量化计算 (每条指令处理 2 个 half)
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; j += 2) {
          // 将 frag_a[i] 复制到 half2 的两个分量
          half2 a2 = __half2half2(frag_a[i]);
          // 将 frag_b[j], frag_b[j+1] 组合成 half2
          half2 b2 = *HALF2(&frag_b[j]);
          // 将 accum[i][j], accum[i][j+1] 组合成 half2
          half2 c2 = *HALF2(&accum[i][j]);
          // FMA: c2 = a2 * b2 + c2
          *HALF2(&accum[i][j]) = __hfma2(a2, b2, c2);
        }
      }
    }

    __syncthreads();
  }

  // ========== 写回结果到全局内存 ==========
  // 使用 float4 向量化存储
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int row = thread_m + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < TN; j += 8) {
        int col = thread_n + j;
        if (col + 7 < N) {
          // 向量化存储 8 个 half
          *FLOAT4(&C[row * N + col]) = *FLOAT4(&accum[i][j]);
        } else {
          // 边界处理: 逐个元素存储
#pragma unroll
          for (int jj = 0; jj < 8; ++jj) {
            if (col + jj < N) {
              C[row * N + col + jj] = accum[i][j + jj];
            }
          }
        }
      }
    }
  }
}
