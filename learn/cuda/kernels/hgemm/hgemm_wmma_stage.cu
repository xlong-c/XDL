#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

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
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_m = warp_id / WARP_TILE_M;
  const int warp_n = warp_id % WARP_TILE_M;

  int NUM_K_TILES = div_ceil(K, WMMA_K);

  constexpr int BM = WMMA_M * WMMA_TILE_M * WMMA_TILE_N; // 16*4*2 = 128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16*2*4 = 128
  constexpr int BK = WMMA_K;                             // 16

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
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 预加载
    int idx_g_ak = k * WMMA_K + idx_s_ak;
    int addr_a = idx_g_ak * K + idx_g_ak;
    int idx_g_bk = k * WMMA_K + idx_s_bk;
    int addr_b = idx_g_bk * N + idx_g_bn;

    uint32_t smem_a_ptr = (smem_a_ptr_base + (K * s_a_offset + idx_s_am * (BK + A_PAD) + idx_s_ak) * sizeof(half));
    uint32_t smem_b_ptr = (smem_b_ptr_base + (N * s_b_offset + idx_s_bn * (BN + B_PAD) + idx_s_bk) * sizeof(half));

    CP_ASYNC_CG(smem_a_ptr, &A[addr_a], 16) // 每个线程8个half
    CP_ASYNC_CG(smem_b_ptr, &B[addr_b], 16)

    // commit
    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(0);
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    int smem_sel = (k + 1) % K_STAGE;
    int smem_sel_next = k % K_STAGE;

    int idx_g_ak = k * WMMA_K + idx_s_ak;
    int addr_a = idx_g_ak * K + idx_g_ak;
    int idx_g_bk = k * WMMA_K + idx_s_bk;
    int addr_b = idx_g_bk * N + idx_g_bn;

    uint32_t smem_a_ptr = (smem_a_ptr_base + (K * s_a_offset + idx_s_am * (BK + A_PAD) + idx_s_ak) * sizeof(half));
    uint32_t smem_b_ptr = (smem_b_ptr_base + (N * s_b_offset + idx_s_bn * (BN + B_PAD) + idx_s_bk) * sizeof(half));

    CP_ASYNC_CG(smem_a_ptr, &A[addr_a], 16)
    CP_ASYNC_CG(smem_b_ptr, &B[addr_b], 16)

    // commit
    CP_ASYNC_COMMIT_GROUP();
    // 读取数据
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_K, wmma::row_major> a_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, wmma::row_major> b_frag[WARP_TILE_N];
    // stage 0
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(
          a_frag[i],
          &s_a[smem_sel][warp_smem_a_m][0],
          BK + A_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_N; ++i) {
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N;
      wmma::load_matrix_sync(
          b_frag[i],
          &s_b[smem_sel][warp_smem_b_n][0],
          BN + B_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i]);
      }
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }
  if ((K_STAGE - 2) > 0) { // 最后一批数据缓存未完成，等待
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  { // stage K_STAGE - 1 最后一节
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) {
      // 选择缓存
      const int stage_sel = (NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE;
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_K, wmma::row_major> a_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, wmma::row_major> b_frag[WARP_TILE_N];
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        wmma ::load_matrix_sync(
            a_frag[i],
            &s_a[stage_sel][warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M][0],
            BK + A_PAD);
      }
#pragma unroll
      for (int i = 0; i < WARP_TILE_N; ++i) {
        wmma ::load_matrix_sync(
            b_frag[i],
            &s_b[stage_sel][warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N][0],
            BN + B_PAD);
      }
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i]);
        }
      }
    }
  }

//store
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; j++) {
      const int store_gmem_am = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_bn = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(
          C + store_gmem_am * N + store_gmem_bn,
          acc[i][j],
          N, wmma::mem_row_major);
    }
  }
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int K_STAGE = 2, const int A_PAD = 0, const int B_PAD = 0,
          const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256) hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel(
    half *A, half *B, half *C,
    int M, int N, int K) {
  const int bx = blockIdx.x + ((int)BLOCK_SWIZZLE) * gridDim.x * blockIdx.z;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  const int BM = WMMA_M * WARP_TILE_M * WMMA_TILE_M;
  const int BN = WMMA_N * WARP_TILE_N * WMMA_TILE_N;
  const int BK = WMMA_K;
  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + (BK + A_PAD) * BM * K_STAGE;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_m = warp_id / WMMA_TILE_N;
  const int warp_n = warp_id % WMMA_TILE_N;

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid & 1) << 3;
  int load_smem_b_k = tid / 16; // 128/8 = 16
  int load_smem_b_n = (tid & 15) << 3;
  int load_gmem_a_m = BM * by + load_smem_a_m;
  int load_gmem_b_n = BN * bx + load_smem_b_n;

  if (load_gmem_a_m >= M || load_gmem_b_n >= N) {
    return;
  }

  wmma ::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[WARP_TILE_M][WARP_TILE_N];
// 双层循环初始化acc
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }
  // 共享内存指针
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

// 缓存池预填充
#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(half)); // 第几个stage,第几行,第几个缓存位置
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
  __syncthreads();
#pragma unroll // 开始计算
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    int smem_sel = (k + 1) % K_STAGE; // 如果K_STAGE = 2,则k = 1时，smem_sel = 0
    int smem_sel_nex = k % K_STAGE;

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k)) *
        sizeof(half);
    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n)) *
        sizeof(half);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half> a_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, half> b_frag[WARP_TILE_N];
#pragma unroll // 加载寄存器
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      // 这里直接加载一行,所以行偏移为0
      half *load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) + 0);
      wmma::load_matrix_sync(a_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
    }
#pragma unroll
    for (int i = 0; i < WARP_TILE_N; ++i) {
      int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N;
      half *load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n);
      wmma::load_matrix_sync(b_frag[i], load_smem_b_frag_ptr, BN + B_PAD);
    }
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
      }
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  { //尾巴一节
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half> a_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, half> b_frag[WARP_TILE_N];
#pragma unroll // 加载寄存器
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        // 这里直接加载一行,所以行偏移为0
        half *load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) + 0);
        wmma::load_matrix_sync(a_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
      }
#pragma unroll
      for (int i = 0; i < WARP_TILE_N; ++i) {
        int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N;
        half *load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n);
        wmma::load_matrix_sync(b_frag[i], load_smem_b_frag_ptr, BN + B_PAD);
      }
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }
  }
  { // 存数据
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        const int store_gmem_am = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        const int store_gmem_an = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::store_matrix_sync(C + store_gmem_am * N + store_gmem_an, acc[i][j], N, wmma::mem_row_major);
      }
    }
  }
}
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void
hgemm_wmma_m16k16n16_mma4x4_warp4x4_stage_dsmem_kernel(const half *A, const half *B, half *C,
                                                       const int M, const int N, const int K) {
  const int by = blockIdx.y;
  const int bx = blockIdx.x + ((int)BLOCK_SWIZZLE) * gridDim.x * blockIdx.z;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  const int BM = WMMA_M * WARP_TILE_M * WMMA_TILE_M;
  const int BN = WMMA_N * WARP_TILE_N * WMMA_TILE_N;
  const int BK = WMMA_K;
  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 每个stage的缓存大小
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);
  const int warp = tid / WARP_SIZE;
  const int warp_m = warp / WMMA_TILE_N;
  const int warp_n = warp % WMMA_TILE_N;
  // BM = 16 x4 x4 = 256 , 256 / 32 = 8
  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid & 1) << 3;
  int load_smem_b_k = tid / 32; //  256 /8 = 32
  int load_smem_b_n = (tid & 31) << 3;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) {
    return;
  }

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[WARP_TILE_M][WARP_TILE_N];
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    uint32_t load_smem_a_ptr =
        smem_a_base_ptr +
        (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half);
    uint32_t load_smem_b_ptr =
        smem_b_base_ptr +
        (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    const int smem_sel = (k + 1) % K_STAGE;
    const int smem_sel_next = k % K_STAGE;

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    uint32_t load_smem_a_ptr =
        smem_a_base_ptr +
        (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
            sizeof(half);
    uint32_t load_smem_b_ptr =
        smem_b_base_ptr +
        (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
            sizeof(half);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      half *load_smem_a_frag_ptr =
          s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD);
      wmma::load_matrix_sync(a_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_N; ++i) {
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N;
      half *load_smem_b_frag_ptr =
          s_b + smem_sel * s_b_stage_offset + warp_smem_b_n;
      wmma::load_matrix_sync(b_frag[i], load_smem_b_frag_ptr, BN + B_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  // 处理流水线尾部尚未计算的 stage。
#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    const int stage_sel = (NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      half *load_smem_a_frag_ptr =
          s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD);
      wmma::load_matrix_sync(a_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_N; ++i) {
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N;
      half *load_smem_b_frag_ptr =
          s_b + stage_sel * s_b_stage_offset + warp_smem_b_n;
      wmma::load_matrix_sync(b_frag[i], load_smem_b_frag_ptr, BN + B_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_m * N + store_gmem_n, acc[i][j], N, wmma::mem_row_major);
    }
  }
}
