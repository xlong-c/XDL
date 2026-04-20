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
