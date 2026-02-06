// #include "cooperative_groups.h"
// #include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
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
  const int idx = (blockIdx.x * NumThreadsPerBlock + threadIdx.x) * 8;
  half reg[8];
  LDST128BITS(reg) = LDST128BITS(x[idx]);
  float max_val = -FLT_MAX;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    max_val = fmaxf(max_val, __half2float(reg[i]));
  }
  max_val = block_reduce_max_f32<NumThreadsPerBlock>(max_val);

  float exp_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    float exp_val = expf(__half2float(reg[i]) - max_val);
    exp_sum += exp_val;
    reg[i] = __float2half(exp_val);
  }
  exp_sum = block_reduce_sum_f32<NumThreadsPerBlock>(exp_sum);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    reg[i] = __float2half(__half2float(reg[i]) / exp_sum);
  }
  if ((idx + 7) < N)
    LDST128BITS(y[idx]) = LDST128BITS(reg);
}

template <int NumThreadsPerBlock>
__global__ void online_safe_softmax_f32_per_token_kernel(const float *x,
                                                         float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NumThreadsPerBlock + tid;
  constexpr int WARP_NUM = NumThreadsPerBlock / WARP_SIZE;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  MD val;
  val.m = idx < N ? x[idx] : -FLT_MAX;
  val.d = idx < N ? 1.0f : 0.0f;
  __shared__ MD smem[WARP_NUM];
  MD res = warp_reduce_md_op(val);
  if (lane_id == 0) {
    smem[warp_id] = res;
  }
  __syncthreads();

  if (idx < WARP_SIZE) {
    MD block_res = smem[lane_id];
    block_res = warp_reduce_md_op(block_res);
    if (lane_id == 0) {
      smem[lane_id] = block_res;
    }
  }
  __syncthreads();
  MD final_res = smem[0];
  float d_total_inverse = __fdividef(1.0f, final_res.d);
  if (idx < N) {
    y[idx] = __expf(x[idx] - final_res.m) * d_total_inverse;
  }
}

template <int NumThreadsPerBlock>
__global__ void online_safe_softmax_f16x8_f32_per_token_kernel(half *x, half *y,
                                                               int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NumThreadsPerBlock + tid) * 8;
  constexpr int WARP_NUM = NumThreadsPerBlock / WARP_SIZE;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  half reg[8];
  if (idx + 7 < N) {
    LDST128BITS(reg) = LDST128BITS(x[idx]);
  }

  half max_val = __hneg(CUDART_MAX_NORMAL_FP16);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    max_val = __hmax(max_val, reg[i]);
  }
  MD val;
  val.m = idx + 7 < N ? __half2float(max_val) : -FLT_MAX;
  val.d = idx + 7 < N ? 1.0f : 0.0f;
  __shared__ MD smem[WARP_NUM];
  MD res = warp_reduce_md_op(val);
  if (lane_id == 0) {
    smem[warp_id] = res;
  }
  __syncthreads();
  if (idx < WARP_SIZE) {
    MD block_res = smem[lane_id];
    block_res = warp_reduce_md_op(block_res);
    if (lane_id == 0) {
      smem[lane_id] = block_res;
    }
  }
  __syncthreads();
  MD final_res = smem[0];
  float d_total_inverse = __fdividef(1.0f, final_res.d);
  if (idx + 7 < N) {
#pragma unroll
    for (int i = 0; i < 8; i++) {

      reg[i] = __float2half(__expf(__half2float(reg[i]) - final_res.m) *
                            d_total_inverse);
    }
    LDST128BITS(y[idx]) = LDST128BITS(reg);
  }
}

void cpu_softmax_f32(const float *x, float *y, int N) {
  float max_val = -FLT_MAX;
  for (int i = 0; i < N; i++) {
    max_val = fmaxf(max_val, x[i]);
  }
  float exp_sum = 0.0f;
  for (int i = 0; i < N; i++) {
    exp_sum += expf(x[i] - max_val);
  }
  for (int i = 0; i < N; i++) {
    y[i] = expf(x[i] - max_val) / exp_sum;
  }
}

void cpu_softmax_f16(const half *x, half *y, int N) {
  float max_val = -FLT_MAX;
  for (int i = 0; i < N; i++) {
    max_val = fmaxf(max_val, __half2float(x[i]));
  }
  float exp_sum = 0.0f;
  for (int i = 0; i < N; i++) {
    exp_sum += expf(__half2float(x[i]) - max_val);
  }
  for (int i = 0; i < N; i++) {
    y[i] = __float2half(expf(__half2float(x[i]) - max_val) / exp_sum);
  }
}

float max_abs_error_f32(const float *ref, const float *result, int N) {
  float max_error = 0.0f;
  for (int i = 0; i < N; i++) {
    float error = fabsf(ref[i] - result[i]);
    max_error = fmaxf(max_error, error);
  }
  return max_error;
}

float max_abs_error_f16(const half *ref, const half *result, int N) {
  float max_error = 0.0f;
  for (int i = 0; i < N; i++) {
    float error = fabsf(__half2float(ref[i]) - __half2float(result[i]));
    max_error = fmaxf(max_error, error);
  }
  return max_error;
}

struct BenchmarkResult {
  const char *name;
  float time_ms;
  float max_error;
};

int main() {
  int N = 1024 * 1024 * 16; // 1M elements
  const int num_iterations = 100;
  const int warmup_iterations = 10;

  printf("=== Softmax Kernel Benchmark ===\n");
  printf("Problem size: %d elements\n", N);
  printf("Iterations: %d (warmup: %d)\n\n", num_iterations, warmup_iterations);

  // Allocate host memory
  float *h_x_f32 = (float *)malloc(N * sizeof(float));
  float *h_y_f32 = (float *)malloc(N * sizeof(float));
  float *h_ref_f32 = (float *)malloc(N * sizeof(float));

  half *h_x_f16 = (half *)malloc(N * sizeof(half));
  half *h_y_f16 = (half *)malloc(N * sizeof(half));
  half *h_ref_f16 = (half *)malloc(N * sizeof(half));

  // Initialize data
  srand(42);
  for (int i = 0; i < N; i++) {
    h_x_f32[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
    h_x_f16[i] = __float2half(h_x_f32[i]);
  }

  // Compute CPU reference
  cpu_softmax_f32(h_x_f32, h_ref_f32, N);
  cpu_softmax_f16(h_x_f16, h_ref_f16, N);

  // Allocate device memory
  float *d_x_f32, *d_y_f32;
  half *d_x_f16, *d_y_f16;
  cudaMalloc(&d_x_f32, N * sizeof(float));
  cudaMalloc(&d_y_f32, N * sizeof(float));
  cudaMalloc(&d_x_f16, N * sizeof(half));
  cudaMalloc(&d_y_f16, N * sizeof(half));

  // Copy data to device
  cudaMemcpy(d_x_f32, h_x_f32, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_f16, h_x_f16, N * sizeof(half), cudaMemcpyHostToDevice);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Define kernel configurations
  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;
  const int num_blocks_f32x4 =
      (N + threads_per_block * 4 - 1) / (threads_per_block * 4);
  const int num_blocks_f16x2 =
      (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
  const int num_blocks_f16x8 =
      (N + threads_per_block * 8 - 1) / (threads_per_block * 8);

  // Benchmark helper for f32 kernels
  auto benchmark_f32_kernel = [&](const char *name, auto kernel, dim3 grid,
                                  dim3 block) {
    void *args[] = {&d_x_f32, &d_y_f32, &N};

    // Warmup
    for (int i = 0; i < warmup_iterations; i++) {
      cudaLaunchKernel((const void *)kernel, grid, block, args, 0, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
      cudaLaunchKernel((const void *)kernel, grid, block, args, 0, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time = elapsed_ms / num_iterations;

    cudaMemcpy(h_y_f32, d_y_f32, N * sizeof(float), cudaMemcpyDeviceToHost);
    float max_error = max_abs_error_f32(h_ref_f32, h_y_f32, N);

    printf("%-40s %8.3f ms  %12.6e\n", name, avg_time, max_error);
  };

  // Benchmark helper for f16 kernels
  auto benchmark_f16_kernel = [&](const char *name, auto kernel, dim3 grid,
                                  dim3 block) {
    void *args[] = {&d_x_f16, &d_y_f16, &N};

    // Warmup
    for (int i = 0; i < warmup_iterations; i++) {
      cudaLaunchKernel((const void *)kernel, grid, block, args, 0, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
      cudaLaunchKernel((const void *)kernel, grid, block, args, 0, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time = elapsed_ms / num_iterations;

    cudaMemcpy(h_y_f16, d_y_f16, N * sizeof(half), cudaMemcpyDeviceToHost);
    float max_error = max_abs_error_f16(h_ref_f16, h_y_f16, N);

    printf("%-40s %8.3f ms  %12.6e\n", name, avg_time, max_error);
  };

  printf("%-40s %10s  %12s\n", "Kernel", "Time(ms)", "Max Error");
  printf("%s\n", std::string(65, '-').c_str());

  // Benchmark f32 kernels
  benchmark_f32_kernel("softmax_f32_per_token",
                       (void *)softmax_f32_per_token_kernel<256>, num_blocks,
                       256);
  benchmark_f32_kernel("softmax_f32x4_per_token",
                       (void *)softmax_f32x4_per_token_kernel<64>,
                       num_blocks_f32x4, 64);
  benchmark_f32_kernel("safe_softmax_f32x4_per_token",
                       (void *)safe_softmax_f32x4_per_token_kernel<64>,
                       num_blocks_f32x4, 64);
  benchmark_f32_kernel("online_safe_softmax_f32_per_token",
                       (void *)online_safe_softmax_f32_per_token_kernel<256>,
                       num_blocks, 256);

  // Benchmark f16 kernels
  benchmark_f16_kernel("safe_softmax_f16_f32_per_token",
                       (void *)safe_softmax_f16_f32_per_token_kernel<256>,
                       num_blocks, 256);
  benchmark_f16_kernel("safe_softmax_f16x2_f32_per_token",
                       (void *)safe_softmax_f16x2_f32_per_token_kernel<256>,
                       num_blocks_f16x2, 256);
  benchmark_f16_kernel("safe_softmax_f16x8_f32_per_token",
                       (void *)safe_softmax_f16x8_f32_per_token_kernel<256>,
                       num_blocks_f16x8, 256);
  benchmark_f16_kernel(
      "online_safe_softmax_f16x8_f32_per_token",
      (void *)online_safe_softmax_f16x8_f32_per_token_kernel<256>,
      num_blocks_f16x8, 256);

  printf("%s\n", std::string(65, '-').c_str());

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_x_f32);
  cudaFree(d_y_f32);
  cudaFree(d_x_f16);
  cudaFree(d_y_f16);
  free(h_x_f32);
  free(h_y_f32);
  free(h_ref_f32);
  free(h_x_f16);
  free(h_y_f16);
  free(h_ref_f16);

  printf("\nBenchmark completed.\n");
  return 0;
}