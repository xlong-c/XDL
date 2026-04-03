#include <__clang_cuda_runtime_wrapper.h>
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <vector>
using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) {
  return (a % b) != 0 ? ((a + b - 1) / b) : (a / b);
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16(const half *A, const half *B, half *C, int M, int N, int K) {
  const int NUM_K_ITERS = div_ceil(K, WMMA_K);
  const int idx_g_am = blockIdx.y * WMMA_M; // 当前块对应的m块索引
  const int idx_g_an = blockIdx.x * WMMA_N;
  if (idx_g_am >= M || idx_g_an >= N) {
    return;
  }
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc;
  wmma::fill_fragment(acc, 0.0f);
#pragma unroll
  for (int k = 0; k < NUM_K_ITERS; ++k) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b;
    wmma::load_matrix_sync(a, A + idx_g_am * K + k * WMMA_K, K);
    wmma::load_matrix_sync(b, B + N * k * WMMA_K + idx_g_an, N);
    wmma::mma_sync(acc, a, b, acc);
    __syncthreads();
  }
  wmma::store_matrix_sync(C + idx_g_am * N + idx_g_an, acc, N, wmma::mem_row_major);
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_m4n2(half *A, half *B, half *C, int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  // const int tx = threadIdx.x;
  // const int ty = threadIdx.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int NUM_K_ITERS = div_ceil(K, WMMA_K);  // 迭代次数
  const int BM = WMMA_M * WMMA_TILE_M;          // 块A的行数, 16x4
  const int BN = WMMA_N * WMMA_TILE_N;          // 块B的列数, 16x2
  const int BK = WMMA_K;                        // 16
  __shared__ half s_a[BM][BK], s_b[WMMA_K][BN]; //  64 x 16, 16 x 32
  const int warp_id = tid / WARP_SIZE;          // 0~7
  const int lane_id = tid % WARP_SIZE;          // 0~31
  const int warp_m = warp_id / 2;
  const int warp_n = warp_id % 2;

  // 加载数据
  // 为什么每线程4数据, 数据量 64 x 16 = 1024, 1024/(8*32) = 4
  const int idx_s_ak = (tid % 4) * 4;
  const int idx_s_am = tid / 4; // 每行16个数据, 4数据每一个线程, 四线程每行
  const int idx_s_bn = (tid % 16) / 2;
  const int idx_s_bk = tid / 16; // 每个线程2个, 一行32个数据, 16线程每行
  const int idx_g_am = by * BM + idx_s_am;
  const int idx_g_bn = bx * BN + idx_s_bn;

  if (idx_g_am >= M || idx_g_bn >= N)
    return;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc;
  wmma::fill_fragment(acc, 0.0f);

#pragma unroll
  for (int k = 0; k < NUM_K_ITERS; ++k) {
    int idx_g_ak = idx_g_am * K + k * WMMA_K;
    int idx_g_bk = k * WMMA_K + idx_g_bn;
    int addr_a = idx_g_ak + idx_g_am * K;
    int addr_b = idx_g_bk * N + idx_g_bn;
    LDST64BITS(s_a[idx_s_am][idx_s_ak]) = LDST64BITS(A[addr_a]);
    LDST32BITS(s_b[idx_s_bk][idx_s_bn]) = LDST32BITS(B[addr_b]);
    __syncthreads();
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b;

    wmma::load_matrix_sync(a, &s_a[warp_m * WMMA_M][0], BK);
    wmma::load_matrix_sync(b, &s_b[0][warp_n * WMMA_N], BN);
    wmma::mma_sync(acc, a, b, acc);
    __syncthreads();
  }
  const int idx_c_m = by * BM + warp_m * WMMA_M; // 每个warp对应的m索引
  const int idx_c_n = bx * BN + warp_n * WMMA_N;
  wmma::store_matrix_sync(C + idx_c_m * N + idx_c_n, acc, N, wmma::mem_row_major);
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_tile(half *A, half *B, half *C, int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int NUM_K_ITERS = div_ceil(K, WMMA_K);
  const int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 128
  const int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 128
  const int BK = WMMA_K;
  __shared__ half s_a[BM][BK], s_b[BK][BN]; // 128 16, 16 128
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_m = warp_id / WMMA_TILE_N;
  const int warp_n = warp_id % WMMA_TILE_N;

  //
  // 数据总量 2048 线程数量256
  int idx_s_am = tid / 2;
  int idx_s_ak = (tid % 2) * 8;
  int idx_s_bk = tid / 16;
  int idx_s_bn = (tid % 16) * 8;

  int idx_g_am = by * BM + idx_s_am;
  int idx_g_bn = bx * BN + idx_s_bn;
  if (idx_g_am >= M || idx_g_bn >= N)
    return;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[WMMA_TILE_M][WARP_TILE_N];
#pragma unroll
  for (int i = 0; i < WMMA_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }

#pragma unroll
  for (int k = 0; k < NUM_K_ITERS; ++k) {
    int idx_g_ak = k * WMMA_K + idx_s_ak;
    int idx_g_bk = k * WMMA_K + idx_s_bk;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_bk * N + idx_g_bn;
    LDST128BITS(s_a[idx_s_am][idx_s_ak]) = LDST128BITS(A[addr_a]);
    LDST128BITS(s_b[idx_s_bk][idx_s_bn]) = LDST128BITS(B[addr_b]);
    __syncthreads();
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b[WARP_TILE_N];
#pragma unroll
    for (int i = 0; i < WMMA_TILE_M; ++i) {
      int idx_a = i * WMMA_M + warp_m * (WMMA_M * WMMA_TILE_M);
      wmma::load_matrix_sync(
          a[i], &s_a[idx_a][0], BK);
    }
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int idx_b = j * WMMA_N + warp_n * (WMMA_N * WMMA_TILE_N);
      wmma::load_matrix_sync(
          b[j], &s_b[0][idx_b], BN);
    }
#pragma unroll
    for (int i = 0; i < WMMA_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < WMMA_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int idx_c_m = by * BM + i * WMMA_M + warp_m * (WMMA_M * WMMA_TILE_M);
      int idx_c_n = bx * BN + j * WMMA_N + warp_n * (WMMA_N * WMMA_TILE_N);
      wmma::store_matrix_sync(C + idx_c_m * N + idx_c_n, acc[i][j], N, wmma::mem_row_major);
    }
  }
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int OFFSET = 0>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(half *A, half *B, half *C, int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int NUM_K_ITERS = div_ceil(K, WMMA_K);
  const int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  const int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
  const int BK = WMMA_K;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_m = warp_id / WMMA_TILE_N;
  const int warp_n = warp_id % WMMA_TILE_N;

  __shared__ half s_a[2][BM][BK], s_b[2][BK][BN];

  // 数据总量 2048 线程数量256, 128 * 16 = 2048
  int idx_s_am = tid / 2;
  int idx_s_ak = (tid % 2) * 8;
  int idx_s_bk = tid / 16; // 128 / 8 = 16
  int idx_s_bn = (tid % 16) * 8;

  int idx_g_am = by * BM + idx_s_am;
  int idx_g_bn = bx * BN + idx_s_bn;
  if (idx_g_am >= M || idx_g_bn >= N)
    return;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[WARP_TILE_M][WARP_TILE_N];
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; j++) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }
  { // 预取
    int idx_g_ak = idx_s_ak;
    int idx_g_bk = idx_s_bk;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_bk * N + idx_g_bn;
    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&s_a[0][idx_s_am][idx_s_ak]);
    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[0][idx_s_bk][idx_s_bn]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[addr_a], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[addr_b], 16);
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();
#pragma unroll
  for (int k = 1; k < NUM_K_ITERS; k++) {
    int smem_sel = (k - 1) & 1; // 当前是上一块的坐标
    int smem_sel_next = k & 1;
    int idx_g_ak = k * WMMA_K + idx_s_ak;
    int idx_g_bk = k * WMMA_K + idx_s_bk;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_bk * N + idx_g_bn;

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&s_a[smem_sel_next][idx_s_am][idx_s_ak]);
    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[smem_sel_next][idx_s_bk][idx_s_bn]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[addr_a], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[addr_b], 16);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b[WARP_TILE_N];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++) {
      int idx_a = i * WMMA_M + warp_m * (WMMA_M * WMMA_TILE_M); //
      wmma::load_matrix_sync(
          a[i], &s_a[smem_sel][idx_a][0], BK);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; j++) {
      int idx_b = j * WMMA_N + warp_n * (WMMA_N * WMMA_TILE_N); //
      wmma::load_matrix_sync(
          b[j], &s_b[smem_sel][0][idx_b], BN);
    }

// 计算
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; j++) {
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  { // 计算最后一段K
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b[WARP_TILE_N];
#pragma unroll // 寄存器初始化
    for (int i = 0; i < WARP_TILE_M; i++) {
      int idx_a = i * WMMA_M + warp_m * (WMMA_M * WMMA_TILE_M);
    }
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; j++) {
      int idx_b = j * WMMA_N + warp_n * (WMMA_N * WMMA_TILE_N);
    }
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; j++) {
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
  }
// 存储结果
#pragma unroll
  for (int i = 0; i < WMMA_TILE_M; i++) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; j++) {
      int idx_c_m = by * BM + i * WMMA_M + warp_m * (WMMA_M * WMMA_TILE_M);
      int idx_c_n = bx * BN + j * WMMA_N + warp_n * (WMMA_N * WMMA_TILE_N);
      wmma::store_matrix_sync(C + idx_c_m * N + idx_c_n, acc[i][j], N, wmma::mem_row_major);
    }
  }
}

