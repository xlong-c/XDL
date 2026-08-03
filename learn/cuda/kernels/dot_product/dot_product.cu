#include "dot_product.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <vector_types.h>

#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__global__ void dot_prod_f32_f32_kernel(float *a, float *b, float *y, int N) {
  constexpr int NUM_THREADS = 256;
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  prod = warp_reduce_sum_f32<WARP_SIZE>(prod);

  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    block_sum = warp_reduce_sum_f32<WARP_SIZE>(block_sum);
    if (lane == 0)
      atomicAdd(y, block_sum);
  }
}

__global__ void dot_prod_f32_f32_kernel_v2(float *a, float *b, float *y,
                                           int N) {
  constexpr int NUM_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 16;
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * NUM_THREADS * ITEMS_PER_THREAD;

  float thread_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int idx = block_offset + i * NUM_THREADS + tid;
    if (idx < N) {
      thread_sum += a[idx] * b[idx];
    }
  }

  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  thread_sum = warp_reduce_sum_f32<WARP_SIZE>(thread_sum);

  if (lane == 0)
    reduce_smem[warp] = thread_sum;

  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    block_sum = warp_reduce_sum_f32<WARP_SIZE>(block_sum);
    if (lane == 0)
      atomicAdd(y, block_sum);
  }
}

__global__ void dot_prod_f32x4_f32_kernel(float *a, float *b, float *y, int N) {
  constexpr int NUM_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 16;
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * NUM_THREADS * ITEMS_PER_THREAD;

  float thread_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD / 4; i++) {
    int idx = block_offset + (i * NUM_THREADS + tid) * 4;
    if (idx + 3 < N) {
      float4 reg_a = LDST128BITS(a[idx]);
      float4 reg_b = LDST128BITS(b[idx]);
      thread_sum += reg_a.x * reg_b.x;
      thread_sum += reg_a.y * reg_b.y;
      thread_sum += reg_a.z * reg_b.z;
      thread_sum += reg_a.w * reg_b.w;
    } else {
      // Handle remaining elements if N is not multiple of 4
      for (int j = 0; j < 4; ++j) {
        if (idx + j < N) {
          thread_sum += a[idx + j] * b[idx + j];
        }
      }
    }
  }

  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  thread_sum = warp_reduce_sum_f32<WARP_SIZE>(thread_sum);

  if (lane == 0)
    reduce_smem[warp] = thread_sum;

  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    block_sum = warp_reduce_sum_f32<WARP_SIZE>(block_sum);
    if (lane == 0)
      atomicAdd(y, block_sum);
  }
}

__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  float reg_sum = __half2float(val);
  return warp_reduce_sum_f32<WARP_SIZE>(reg_sum);
}

__global__ void dot_prod_f16_f32_kernel(half *a, half *b, half *y, int mask) {
  constexpr int NUM_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 16; // Each thread processes 16 * 8 = 128 elements
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * NUM_THREADS * ITEMS_PER_THREAD * 8;
  float thread_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int idx = block_offset + (i * NUM_THREADS + tid) * 8;
    if (idx + 7 < mask) {
      half reg_a[8];
      half reg_b[8];
      LDST128BITS(reg_a) = LDST128BITS(a[idx]);
      LDST128BITS(reg_b) = LDST128BITS(b[idx]);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        thread_sum += __half2float(reg_a[j]) * __half2float(reg_b[j]);
      }
    } else {
      for (int j = 0; j < 8; ++j) {
        if (idx + j < mask) {
          thread_sum += __half2float(a[idx + j]) * __half2float(b[idx + j]);
        }
      }
    }
  }

  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  thread_sum = warp_reduce_sum_f32<WARP_SIZE>(thread_sum);

  if (lane == 0)
    reduce_smem[warp] = thread_sum;

  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    block_sum = warp_reduce_sum_f32<WARP_SIZE>(block_sum);
    if (lane == 0) {
      atomicAdd(y, __float2half(block_sum));
    }
  }
}

__global__ void dot_prod_bf16x8_bf16_kernel(__nv_bfloat16 *a, __nv_bfloat16 *b,
                                            __nv_bfloat16 *y, int N) {
  constexpr int NUM_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 16;
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * NUM_THREADS * ITEMS_PER_THREAD * 8;

  float thread_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = block_offset + (i * NUM_THREADS + tid) * 8;
    if (idx + 7 < N) {
      __nv_bfloat16 reg_a[8];
      __nv_bfloat16 reg_b[8];
      LDST128BITS(reg_a) = LDST128BITS(a[idx]);
      LDST128BITS(reg_b) = LDST128BITS(b[idx]);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        thread_sum += __bfloat162float(reg_a[j]) * __bfloat162float(reg_b[j]);
      }
    } else {
      for (int j = 0; j < 8; ++j) {
        if (idx + j < N) {
          thread_sum += __bfloat162float(a[idx + j]) * __bfloat162float(b[idx + j]);
        }
      }
    }
  }
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  thread_sum = warp_reduce_sum_f32<WARP_SIZE>(thread_sum);

  if (lane == 0)
    reduce_smem[warp] = thread_sum;

  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    block_sum = warp_reduce_sum_f32<WARP_SIZE>(block_sum);
    if (lane == 0) {
      atomicAdd(y, __float2bfloat16(block_sum));
    }
  }
}