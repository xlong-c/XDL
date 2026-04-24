#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda.h>
#include <cuda/barrier>

#define WARP_SIZE 32
#define WARPGROUP_SIZE 128
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
// smem descriptor encode for wgmma (from fast.cu)
#define SMEM_DESC_ENCODE(x) ((((uint64_t)(x)) & 0x3FFFF) >> 0x4)
// wgmma fence/commit/wait
#define WGMMA_FENCE() \
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory")
#define WGMMA_COMMIT_GROUP() \
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory")
#define WGMMA_WAIT_GROUP(n) \
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(n) : "memory")

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda ::device::experimental;

HOST_DEVICE_INLINE
int div_ceil(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

DEVICE_INLINE
uint32_t make_smem_desc(half *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0x0000000000000000;
  desc |= SMEM_DESC_ENCODE(addr);
  desc |= SMEM_DESC_ENCODE((uint64_t)16) << 16;
  desc |= SMEM_DESC_ENCODE((uint64_t)1024) << 32;
  desc |= 1llu << 62;
  return desc;
}
#define WGMMA_M64N128K16_F16F16F16(d, sA, sB, ScaleD, ScaleA, ScaleB,         \
                                   TransA, TransB)                            \
  {                                                                           \
    uint64_t desc_a = make_smem_desc(&(sA)[0]);                               \
    uint64_t desc_b = make_smem_desc(&(sB)[0]);                               \
    asm volatile(                                                             \
        "{\n"                                                                 \
        "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "                \
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "                   \
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "                   \
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "                   \
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"                    \
        " %32,"                                                               \
        " %33,"                                                               \
        " %34, %35, %36, %37, %38;\n"                                         \
        "}\n"                                                                 \
        : "+r"((d)[0][0]), "+r"((d)[0][1]), "+r"((d)[0][2]), "+r"((d)[0][3]), \
          "+r"((d)[1][0]), "+r"((d)[1][1]), "+r"((d)[1][2]), "+r"((d)[1][3]), \
          "+r"((d)[2][0]), "+r"((d)[2][1]), "+r"((d)[2][2]), "+r"((d)[2][3]), \
          "+r"((d)[3][0]), "+r"((d)[3][1]), "+r"((d)[3][2]), "+r"((d)[3][3]), \
          "+r"((d)[4][0]), "+r"((d)[4][1]), "+r"((d)[4][2]), "+r"((d)[4][3]), \
          "+r"((d)[5][0]), "+r"((d)[5][1]), "+r"((d)[5][2]), "+r"((d)[5][3]), \
          "+r"((d)[6][0]), "+r"((d)[6][1]), "+r"((d)[6][2]), "+r"((d)[6][3]), \
          "+r"((d)[7][0]), "+r"((d)[7][1]), "+r"((d)[7][2]), "+r"((d)[7][3])  \
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),                     \
          "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),                         \
          "n"(int32_t(TransA)), "n"(int32_t(TransB)));                        \
  }
template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map,
                                              half *gmem_ptr,
                                              int blocks_height,
                                              int blocks_width) {
  void *gmem_address = (void *)gmem_ptr;
  uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize * blocks_width,
                                 (uint64_t)BlockMajorSize * blocks_height,
                                 1, 1, 1};
  uint64_t gmem_prob_stride[5] = {
      sizeof(half), sizeof(half) * BlockMinorSize * blocks_width, 0, 0, 0};
  uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize),
                                uint32_t(BlockMajorSize), 1, 1, 1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};
  CUresult result = cuTensorMapEncodeTiled(
      tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmem_address,
      gmem_prob_shape, gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS)
    printf("cuTensorMapEncodeTiled failed: %d\n", (int)result);
}
__host__ static inline CUtensorMap *allocate_and_create_tensor_map(
    half *src, int blocks_height, int blocks_width) {
  CUtensorMap *tma_map_d;
  cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
  CUtensorMap tma_map_host;
  create_tensor_map<128, 64>(&tma_map_host, src, blocks_height, blocks_width);
  cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap),
             cudaMemcpyHostToDevice);
  return tma_map_d;
}

template <int BM, int BN, int BK, int QSIZE>
struct WgmmaSMem {
  alignas(128) half A[BM * BK * QSIZE];
  alignas(128) half B[BK * BN * QSIZE];
};

// TN: A row major MxK, B col major NxK, C row major MxN
// 128x128, wgmma m64n128k16, warp specialized (1 producer + 1 consumer),
// stages, block swizzle, TMA, f16 accum
template <const int WGMMA_M = 64, const int WGMMA_N = 128,
          const int WGMMA_K = 16, const int BM = 128, const int BN = 128,
          const int BK = 64, const int NUM_THREADS = 256,
          const int K_STAGE = 3, const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_wgmma_m64n128k16_f16acc_stages_tma_ws_tn_kernel(
        int M, int N, int K,
        const CUtensorMap *__restrict__ tensorMapA, const CUtensorMap *__restrict__ tensorMapB) {
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  constexpr int num_consumers = (NUM_THREADS / WARPGROUP_SIZE) - 1;
  constexpr int B_WG_M = BM / num_consumers;
  if (bx >= div_ceil(N, BN) || by >= div_ceil(M, BM))
    return;
  extern __shared__ __align__(128) uint8_t smem[];
  WgmmaSMem<BM, BN, BK, K_STAGE> &s = *reinterpret_cast<WgmmaSMem<BM, BN, BK, K_STAGE> *>(smem);
  half *s_a = s.A;
  half *s_b = s.B;
#pragma nv_diag_suppress static_val_with_dynamic_init
  __shared__ barrier full[K_STAGE], empty[K_STAGE];

  const int num_blocks_k = div_ceil(K, BK);
  const in
}
