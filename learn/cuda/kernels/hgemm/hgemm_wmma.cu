#include <__clang_cuda_runtime_wrapper.h>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
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
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b;
    wmma::load_matrix_sync(a, A + idx_g_am * K + k * WMMA_K, K);
    wmma::load_matrix_sync(b, B + N * k * WMMA_K + idx_g_an, N);
    wmma::mma_sync(acc, a, b, acc);
    __syncthreads();
  }
  wmma::store_matrix_sync(C + idx_g_am * N + idx_g_an, acc, N, wmma::mem_row_major);
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_tiled_m4n2(half *A, half *B, half *C, int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_ITERS = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M; // 16 * 4 = 64
  constexpr int BN = WMMA_N * WMMA_TILE_N; // 16 * 2 = 32
  constexpr int BK = WMMA_K;
  __shared__ half s_a[BM][BK];
  __shared__ half s_b[BK][BN];
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_m = warp_id / 4;
  const int warp_n = warp_id % 2;
  //每个线程读取的数据, 16*16/32=4
  const int load_smem_a_m = tid / 4; //s_a 64x16 一次4个数据,一行4个线程
  const int load_smem_a_k = (tid % 4) * 4;
  const int load_smem_b_k = tid / 16; // s_b 16x32 一次2个数据,一行16个线程
  const int load_smem_b_n = (tid % 16) * 2;
  const int load_gmem_a_m = by * BM + load_smem_a_m;
  const int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m > M || load_gmem_b_n > N)
    return;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[WMMA_TILE_M][WMMA_TILE_N];
#pragma unroll
  for (int i = 0; i < WMMA_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WMMA_TILE_N; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }
#pragma unroll
  for (int k = 0; k < NUM_K_ITERS; ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int addr_a = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int addr_b = load_gmem_b_k * N + load_gmem_b_n;
    LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST128BITS(B[addr_b]);
    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST128BITS(A[addr_a]);
    __syncthreads();
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b;
#pragma unroll
    for (int i = 0; i < WMMA_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * WMMA_M * WMMA_TILE_M + i * WMMA_M;
      wmma::load_matrix_sync(a, &s_a[warp_smem_a_m][0], BK);
    }
  }
}