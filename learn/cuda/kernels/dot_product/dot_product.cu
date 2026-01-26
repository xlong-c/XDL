#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>

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