#include "cooperative_groups.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cfloat>
#include <cmath>
#include <vector_types.h>
#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

#ifndef FLT_MAX
#define FLT_MAX __FLT_MAX__
#endif

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
  for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, stride);
  }
  return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
  for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
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
  value = (lane < NumWarps) ? smem[lane] : 0.0f;
  value = warp_reduce_sum_f32<WARP_SIZE>(value);
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template <const int NumThreadsPerBlock>
__device__ float block_reduce_max_f32(float val) {
  constexpr int NumWarps = (NumThreadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float smem[NumWarps];
  float value = warp_reduce_max_f32<WARP_SIZE>(val);
  if (lane == 0)
    smem[warp] = value;
  __syncthreads();
  value = (lane < NumWarps) ? smem[lane] : 0.0f;
  value = warp_reduce_max_f32<WARP_SIZE>(value);
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template <const int NumThreadsPerBlock = 256>
__global__ void softmax_f32_per_token_kernel(float *x, float *y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * NumThreadsPerBlock + tid;

  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NumThreadsPerBlock>(exp_val);

  if (idx < N)
    y[idx] = exp_val / exp_sum;
}

template <const int NumThreadsPerBlock = 256 / 4>
__global__ void softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * NumThreadsPerBlock + tid) * 4;

  // 避免分支发散：所有线程都加载数据，无效位置设为0
  float4 reg_x = {0.0f, 0.0f, 0.0f, 0.0f};
  if (idx < N) {
    reg_x = LDST128BITS(x[idx]);
  }

  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;

  float exp_val = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;
  float exp_sum = block_reduce_sum_f32<NumThreadsPerBlock>(exp_val);

  // 只在有效位置写回结果
  if (idx + 0 < N)
    y[idx + 0] = reg_exp.x / exp_sum;
  if (idx + 1 < N)
    y[idx + 1] = reg_exp.y / exp_sum;
  if (idx + 2 < N)
    y[idx + 2] = reg_exp.z / exp_sum;
  if (idx + 3 < N)
    y[idx + 3] = reg_exp.w / exp_sum;
}

// softmax
//  math format : y = exp(x) / sum(exp(x))
//  safe softmax
//  math format : y = exp(x - max(x)) / sum(exp(x - max(x)))

template <const int NumThreadsPerBlock = 256 / 4>
__global__ void safe_softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * NumThreadsPerBlock + tid) * 4;

  // 避免分支发散：所有线程都加载数据，无效位置设为 -FLT_MAX
  float4 reg_x = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
  if (idx < N) {
    reg_x = LDST128BITS(x[idx]);
  }

  reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
  reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
  reg_x.z = (idx + 2 < N) ? reg_x.z : -FLT_MAX;
  reg_x.w = (idx + 3 < N) ? reg_x.w : -FLT_MAX;

  float max_val = fmaxf(reg_x.x, reg_x.y);
  max_val = fmaxf(max_val, reg_x.z);
  max_val = fmaxf(max_val, reg_x.w);
  max_val = block_reduce_max_f32<NumThreadsPerBlock>(max_val);

  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y - max_val) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z - max_val) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w - max_val) : 0.0f;
  float exp_val = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;
  float exp_sum = block_reduce_sum_f32<NumThreadsPerBlock>(exp_val);

  // 只在有效位置写回结果
  if (idx + 0 < N)
    y[idx + 0] = reg_exp.x / exp_sum;
  if (idx + 1 < N)
    y[idx + 1] = reg_exp.y / exp_sum;
  if (idx + 2 < N)
    y[idx + 2] = reg_exp.z / exp_sum;
  if (idx + 3 < N)
    y[idx + 3] = reg_exp.w / exp_sum;
}

template <const int NumThreadsPerBlock = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half *x, half *y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * NumThreadsPerBlock + tid;

  // 使用三元运算符避免分支发散
  float x_val = (idx < N) ? __half2float(x[idx]) : -FLT_MAX;
  float max_val = block_reduce_max_f32<NumThreadsPerBlock>(x_val);
  float exp_val = (idx < N) ? expf(x_val - max_val) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NumThreadsPerBlock>(exp_val);

  // 使用三元运算符避免分支发散
  if (idx < N) {
    y[idx] = __float2half(exp_val / exp_sum);
  }
}

template <const int NumThreadsPerBlock = 256>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half *x, half *y,
                                                        int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * NumThreadsPerBlock + tid) * 2;

  // 避免分支发散：所有线程都加载数据，无效位置设为 -FLT_MAX
  float2 reg_x = __half22float2(HALF2(x[idx]));
  float max_val = -FLT_MAX;
  max_val = ((idx + 0 < N)) ? fmaxf(max_val, reg_x.x) : -FLT_MAX;
  max_val = ((idx + 1 < N)) ? fmaxf(max_val, reg_x.y) : -FLT_MAX;
  // 处理边界：无效位置设为 -FLT_MAX（不影响 max 计算）
  max_val = block_reduce_max_f32<NumThreadsPerBlock>(max_val);

  // 计算 exp(x - max)
  float2 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y - max_val) : 0.0f;

  float exp_val = reg_exp.x + reg_exp.y;
  float exp_sum = block_reduce_sum_f32<NumThreadsPerBlock>(exp_val);

  // 只在有效位置写回结果
  if (idx + 1 < N)
    y[idx + 1] = __float2half(reg_exp.y / exp_sum);
}

template <const int NumThreadsPerBlock = 256>
__global__ void safe_softmax_f16x8_f32_per_token_kernel(half *x, half *y,
                                                        int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * NumThreadsPerBlock + threadIdx.x;
}