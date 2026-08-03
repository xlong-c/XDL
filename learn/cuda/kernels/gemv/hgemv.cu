#include "algorithm"
#include "cuda_bf16.h"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat2 *>(&(value))[0])

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

__global__ void hgemv_k32_f16_kernel(half *a, half *x, half *y, int M, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;
  if (m < M) {
    half sum = CUDART_ZERO_FP16;
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; w++) {
      int k = w * WARP_SIZE + lane;
      sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum_f16(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

__device__ __forceinline__ int up_div(int a, int b) { return (a + b - 1) / b; }

__global__ void hgemv_k128_f16x4_kernel(half *a, half *x, half *y, int M,
                                        int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;
  if (m < M) {
    half sum = CUDART_ZERO_FP16;
    int NUM_WARPS = up_div(up_div(K, WARP_SIZE), 4);
#pragma unroll
    for (int w = 0; w < NUM_WARPS; w++) {
      int k = (w * WARP_SIZE + lane) * 4;
      half2 reg_x_0 = HALF2(x[k + 0]);
      half2 reg_x_1 = HALF2(x[k + 2]);
      half2 reg_a_0 = HALF2(a[m * K + k + 0]);
      half2 reg_a_1 = HALF2(a[m * K + k + 1]);
      sum += reg_a_0.x * reg_x_0.x;
      sum += reg_a_0.y * reg_x_0.y;
      sum += reg_a_1.x * reg_x_1.x;
      sum += reg_a_1.y * reg_x_1.y;
    }
    sum = warp_reduce_sum_f16(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

template <const int ROW_PER_WARP = 2>
__global__ void hgemv_k16_f16_kernel(half *a, half *x, half *y, int M, int K) {
  constexpr int K_WARP_SIZE = up_div(WARP_SIZE, ROW_PER_WARP);
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = threadIdx.x % WARP_SIZE;
  int k = lane % K_WARP_SIZE;
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M) {
    half sum = a[m * K + k] * x[k];
    sum = warp_reduce_sum_f16<K_WARP_SIZE>(sum);
    if (k == 0)
      y[m] = sum;
  }
}
