#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_kernel(float *x, float *y, int mask) {
  int tid = threadIdx.x;
  int idx = NUM_THREADS * blockIdx.x + threadIdx.x;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  float sum = (idx < mask) ? x[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_kernel(float *x, float *y,
                                                  int mask) {
  int tid = threadIdx.x;
  int idx = NUM_THREADS * blockIdx.x + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_x = FLOAT4(x[idx]);
  float sum = (idx < mask) ? (reg_x.x + reg_x.y + reg_x.z + reg_x.w) : 0.0f;
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// FP16
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f32(half val) {
  float val_f32 = __half2float(val);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return __float2half(val_f32);
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_f16_f16_kernel(half *x, half *y, int mask) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ half reduce_smem[NUM_WARPS];
  half sum = (idx < mask) ? x[idx] : __float2half(0.0f);

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  sum = warp_reduce_sum_f16_f16<WARP_SIZE>(sum);

  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();

  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);

  if (warp == 0)
    sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);

  if (tid == 0)
    atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half *x, half *y,
                                                      int mask) {
  int tid = threadIdx.x;
  int idx = (blockDim.x * NUM_THREADS + tid) * 2;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float smem[NUM_WARPS];
  half2 reg_x = HALF2(x[idx]);
  half sum = (idx < mask) ? __hadd(reg_x.x, reg_x.y) : CUDART_ONE_FP16;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum = warp_reduce_sum_f16_f32(sum);
  if (lane == 0) {
    smem[warp] = sum;
  }
  __syncthreads();
  sum = (lane < NUM_WARPS) ? smem[lane] : CUDART_ZERO_FP16;
  if (warp == 0) {
    sum = warp_reduce_sum_f16_f32(sum);
  }
  if (tid == 0) {
    atomicAdd(y, sum);
  }
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_f16x2_f16_kernel(half *x, half *y, int mask) {
  int tid = threadIdx.x;
  int idx = (blockDim.x * NUM_THREADS + tid) * 2;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float smem[NUM_WARPS];
  half2 reg_x = HALF2(x[idx]);
  half sum = idx < mask ? __hadd(reg_x.x, reg_x.y) : CUDART_ZERO_FP16;
  sum = warp_reduce_sum_f16_f16(sum);
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) {
    smem[warp_id] = sum;
  }
  __syncthreads();
  // 满载线程是1024,第一个warp的32线程正好可以满载32个warp的结果
  if (warp_id > 0)
    return;
  sum = (tid < NUM_WARPS) ? smem[tid] : CUDART_ZERO_FP16;
  sum = warp_reduce_sum_f16_f16(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_f16x8_pack_f16_kernel(half *x, half *y,
                                                       int mask) {

  int tid = threadIdx.x;
  int idx = (blockDim.x * blockIdx.x + tid) * 8;
  if (idx > mask)
    return;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float smem[NUM_WARPS];
  half reg_x[8];
  LDST128BITS(reg_x) = LDST128BITS(y[idx]);
  half sum;
  for (int i = 0; i < 8; i++) {
    sum = __hadd(sum, reg_x[i]);
  }
  sum = warp_reduce_sum_f16_f16(sum);
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  smem[warp_id] = sum;
  __syncthreads();
  if (tid > WARP_SIZE)
    return;
  sum = warp_id < NUM_WARPS ? smem[tid] : CUDART_ZERO_FP16;
  sum = warp_reduce_sum_f16_f16(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}