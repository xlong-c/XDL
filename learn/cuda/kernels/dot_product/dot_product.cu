#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
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

template <const int NUM_THREADS = 256>
__global__ void dot_prod_f32_f32_kernel(float *a, float *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

template <const int NUM_THREADS = 256, const int ITEMS_PER_THREAD = 16>
__global__ void dot_prod_f32_f32_kernel_v2(float *a, float *b, float *y,
                                           int N) {
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * NUM_THREADS * ITEMS_PER_THREAD;

  float thread_sum = 0.0f;
// Each thread processes ITEMS_PER_THREAD elements to increase ILP and
// utilization. Using a stride of NUM_THREADS to maintain memory coalescing.
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

  // perform warp sync reduce on the accumulated thread_sum.
  thread_sum = warp_reduce_sum_f32<WARP_SIZE>(thread_sum);

  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = thread_sum;

  __syncthreads(); // make sure the data is in shared memory.

  // the first warp compute the final sum.
  float block_sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    block_sum = warp_reduce_sum_f32<NUM_WARPS>(block_sum);

  if (tid == 0)
    atomicAdd(y, block_sum);
}

// 一个block处理的程序: NUM_THREADS * ITEMS_PER_THREAD
template <const int NUM_THREADS = 256, const int ITEMS_PER_THREAD = 16>
__global__ void dot_prod_f32x4_f32_kernel(float *a, float *b, float *y, int N) {
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
    }
  }
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  thread_sum = warp_reduce_sum_f32(thread_sum);
  if (lane > 0)
    return;
  reduce_smem[warp] = thread_sum;

  __syncthreads();

  float block_sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp > 0)
    return;
  block_sum = warp_reduce_sum_f32(block_sum);

  if (tid > 0)
    return;
  atomicAdd(y, block_sum);
}

__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 1; offset >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  float reg_sum = __half2float(val);
#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 1; offset >>= 1) {
    reg_sum += __half2float(__shfl_xor_sync(0xffffffff, val, offset));
  }
  return reg_sum;
}

template <const int NUM_THREADS = 256, const int ITEMS_PER_THREAD = 16>
__global__ void dot_prod_f16_f32_kernel(half *a, half *b, half *y, int mask) {
  int tid = threadIdx.x;
  int block_offset = NUM_THREADS * blockIdx.x ;


}
