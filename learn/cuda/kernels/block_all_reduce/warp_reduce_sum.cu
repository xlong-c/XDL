#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <vector>

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32 Warp Reduce
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Atomic version
template <const int NUM_THREADS = 256 / 4>
__global__ void reduce_atomic_f32x4_kernel(float *a, float *y, int N) {
  int tid = threadIdx.x;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  float sum = 0.0f;

  for (int i = (blockIdx.x * NUM_THREADS + tid) * 4; i < N;
       i += blockDim.x * gridDim.x * 4) {
    float4 reg_a = FLOAT4(a[i]);
    sum += reg_a.x + reg_a.y + reg_a.z + reg_a.w;
  }

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
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float *a, float *y,
                                                      int N) {
  int tid = threadIdx.x;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float sum = 0.0f;
  // Grid-stride loop: ensures all elements are processed
  for (int i = (blockIdx.x * NUM_THREADS + tid) * 4; i < N;
       i += blockDim.x * gridDim.x * 4) {
    float4 reg_a = FLOAT4(a[i]);
    sum += reg_a.x + reg_a.y + reg_a.z + reg_a.w;
  }

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 4>
__global__ void reduce_atomic_f32x4_kernel_modify(float *a, float *y, int N) {
  int tid = threadIdx.x;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  float sum = 0.0f;
  for (int i = (blockIdx.x * NUM_THREADS + tid) * 4; i < N;
       i += blockDim.x * gridDim.x * 4) {
    float4 reg_a = FLOAT4(a[i]);
    sum += reg_a.x + reg_a.y + reg_a.z + reg_a.w;
  }

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

int main() {
  const int S = 4096*2;
  const int K = 4096*2;
  const int N = S * K;
  const int grid_size = 2048;

  float *d_a, *d_y, *d_partial;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_y, sizeof(float));
  cudaMalloc(&d_partial, grid_size * sizeof(float));

  std::vector<float> h_a(N, 1.0f);
  cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int iters = 1000;
  float result_atomic = 0;

  // --- Case 1: AtomicAdd ---
  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    cudaMemset(d_y, 0, sizeof(float));
    reduce_atomic_f32x4_kernel<256 / 4><<<grid_size, 256 / 4>>>(d_a, d_y, N);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaMemcpy(&result_atomic, d_y, sizeof(float), cudaMemcpyDeviceToHost);

  float ms_atomic = 0;
  cudaEventElapsedTime(&ms_atomic, start, stop);
  std::cout << "f32x4 + Atomic    Avg: " << ms_atomic / iters
            << " ms, Result: " << result_atomic << std::endl;

  // --- Case 2: AtomicAdd Modify ---
  float result_modify = 0;
  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    cudaMemset(d_y, 0, sizeof(float));
    block_all_reduce_sum_f32x4_f32_kernel<256 / 4>
        <<<grid_size, 256 / 4>>>(d_a, d_y, N);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaMemcpy(&result_modify, d_y, sizeof(float), cudaMemcpyDeviceToHost);

  float ms_modify = 0;
  cudaEventElapsedTime(&ms_modify, start, stop);
  std::cout << "f32x4 + Modify    Avg: " << ms_modify / iters
            << " ms, Result: " << result_modify << std::endl;

  cudaFree(d_a);
  cudaFree(d_y);
  cudaFree(d_partial);
  return 0;
}