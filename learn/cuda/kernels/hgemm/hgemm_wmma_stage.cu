#include <__clang_cuda_runtime_wrapper.h>
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

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

#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes));
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes));
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n");
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n");

HOST_DEVICE_INLINE
int div_ceil(int a, int b) {
  return (a % b == 0) ? (a / b) : (a + b - 1) / b;
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel(half *A, half *B, half *C,
                                                                  int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_m = warp_id / WARP_TILE_N;
  const int warp_n = warp_id % WARP_TILE_N;

  int NUM_K_ITERS = div_ceil(K, WMMA_K);

  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16*2*4 = 128
  constexpr int BK = WMMA_K;

  __shared__ half s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD];
  constexpr int s_a_offset = BM + B_PAD;
  constexpr int s_b_offset = BK + A_PAD;

  // 数据索引
  int idx_s_am = tid / 2;
  int idx_s_ak = (tid % 2) * 8;
  int idx_s_bk = tid / 16;
  int idx_s_bn = (tid % 16) * 8;

  int idx_g_am = BM * by + idx_s_am;
  int idx_g_bn = BN * bx + idx_s_bn;
  if (idx_g_am >= M || idx_g_bn >= N)
    return;
  wmma ::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[WARP_TILE_M][WARP_TILE_N];

#pragma unroll // 累加初始填充零
  for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; j++) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }
  // 异步加载smem数据
  uint32_t smem_a_ptr_base = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_ptr_base = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    int idx_g_ak = k * WMMA_K + idx_s_ak;
    int addr_a = idx_g_ak * K + idx_g_ak;
    int idx_g_bk = k * WMMA_K + idx_s_bk;
    int addr_b = idx_g_bk * N + idx_g_bn;

    uint32_t smem_a_ptr = (smem_a_ptr_base + (K * s_a_offset + idx_s_am * (BK + A_PAD) + idx_s_ak) * sizeof(half));
    uint32_t smem_b_ptr = ;
  }
}