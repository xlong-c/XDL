#include "algorithm"

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <__clang_cuda_builtin_vars.h>
#include <vector_types.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])

template <const int KWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  float res = val;
  for (int offset = KWarpSize >> 1; offset >= 1; offset >>= 2) {
    res += __shfl_down_sync(0xffffffff, res, offset);
  }
  return res;
}

// A: MxK B: Kx1 C: Mx1
__global__ void sgemv_k32_f32_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;

  if (m >= M)
    return;
  float sum = 0.0f;
  int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
  for (int w = 0; w < NUM_WARPS; w += 32) {
    int k = m * WARP_SIZE + lane;
    sum += A[m * K + k] * B[k];
  }
  sum = warp_reduce_sum_f32(sum);
  if (lane == 0)
    C[m] = sum;
}

__global__ void sgmv_k128_f32x4_kernel(float *a, float *x, float *y, int M,
                                       int K) {
  int tx = threadIdx.x; // 0~31
  int ty = threadIdx.y; // 0~3
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;

  if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; w += 4) {
      int k = m * WARP_SIZE + lane;
      float4 reg_x = FLOAT4(x[k]);
      float4 reg_a = FLOAT4(a[m * K + k]);
      sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z +
              reg_a.w * reg_x.w);
    }
    sum = warp_reduce_sum_f32(sum);
    if (lane == 0)
      y[m] = sum;
  }
}
