#include "cooperative_groups.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

struct __align__(8) MD {
  float m;
  float d;
};

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  unsigned int mask = 0xffffffff;
#pragma unroll
  for (int stride = kWarpSize >> 1; stride > 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(mask, value.m, stride);
    other.d = __shfl_xor_sync(mask, value.d, stride);
    bool value_bigger = (value.m > other.m);
    MD bigger_m = value_bigger ? value : other;
    MD smaller_m = value_bigger ? value : other;
    value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    value.m = bigger_m.m;
  }
  return value;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int stride = kWarpSize >> 1; stride > 1; stride >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, stride);
  }
  return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
  for (int stride = kWarpSize >> 1; stride > 1; stride >>= 1) {
    float other = __shfl_xor_sync(0xffffffff, val, stride);
    val = fmaxf(val, other);
  }
  return val;
}

template <const int NumThreadsPerBlock>
__device__ float block_reduce_sum_f32(float val) {
  constexpr int NumWarps = (NumThreadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float smem[NumWarps];
  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0)
    smem[warp] = value;
  __syncthreads();
}
