#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
/*
sigmod:
    f(x) = 1/(1+e^(-x))
*/

#define WARP_SIZE 32
#define E 2.718281828459045

// Half precision constants
#define HALF_ONE __float2half(1.0f)

// Bfloat16 constants
#define BF16_ONE __float2bfloat16(1.0f)

// 界限检查
#define MAX_EXP_F32 88.37626647949f
#define MIN_EXP_F32 -88.37626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
#define MAX_EXP_BF16 __float2bfloat16(88.37626647949f)
#define MIN_EXP_BF16 __float2bfloat16(-88.37626647949f)

__device__ float limit_check_f32(float x) {
  return fminf(fminf(x, MIN_EXP_F32), MAX_EXP_F32);
}

__device__ half limit_check_f16(half x) {
  return __hmin(__hmax(x, MIN_EXP_F16), MAX_EXP_F16);
}

// bfloat16 support requires CUDA 11.0+
__device__ __nv_bfloat16 limit_check_bf16(__nv_bfloat16 x) {
  return __hmin(__hmax(x, MIN_EXP_BF16), MAX_EXP_BF16);
}

__global__ void sigmod_f32(float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = -fminf(fmaxf(input[idx], MIN_EXP_F32), MAX_EXP_F32);
    output[idx] = 1.0f / (1.0f + expf(x));
  }
}

__global__ void sigmod_f16(half *input, half *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    half x = __hneg(limit_check_f16(input[idx]));
    output[idx] = __hdiv(HALF_ONE, __hadd(HALF_ONE, hexp(x)));
  }
}

__global__ void sigmod_bf16(__nv_bfloat16 *input, __nv_bfloat16 *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // 步骤1：保持bf16原生计算，无类型转换
    __nv_bfloat16 x = __hneg(limit_check_bf16(input[idx]));
    
    // 步骤2：使用CUDA原生bf16指数函数hexp，直接返回bf16，无需fp32转换
    __nv_bfloat16 exp_bf16 = hexp(x);  // 替代：expf(__bfloat162float(x)) + __float2bfloat16()
    
    // 步骤3：继续使用bf16原生算术指令，完成Sigmoid计算
    output[idx] = __hdiv(BF16_ONE, __hadd(BF16_ONE, exp_bf16));
  }
}

int main() {
  const int N = 1024 * 1024 * 10; // 10M elements
  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;
  const int warmup = 5;
  const int iterations = 100;

  // Host memory allocation
  float *h_input = new float[N];
  float *h_output_f32 = new float[N];
  float *h_output_f16 = new float[N];
  float *h_output_bf16 = new float[N];
  half *h_input_f16 = new half[N];
  half *h_output_f16_half = new half[N];
  __nv_bfloat16 *h_input_bf16 = new __nv_bfloat16[N];
  __nv_bfloat16 *h_output_bf16_half = new __nv_bfloat16[N];

  // Initialize input data
  for (int i = 0; i < N; i++) {
    h_input[i] = (float)i / N - 0.5f; // Range: -0.5 to 0.5
    h_input_f16[i] = __float2half(h_input[i]);
    h_input_bf16[i] = __float2bfloat16(h_input[i]);
  }

  // Device memory allocation (float32)
  float *d_input_f32, *d_output_f32;
  cudaMalloc(&d_input_f32, N * sizeof(float));
  cudaMalloc(&d_output_f32, N * sizeof(float));

  // Device memory allocation (float16)
  half *d_input_f16, *d_output_f16;
  cudaMalloc(&d_input_f16, N * sizeof(half));
  cudaMalloc(&d_output_f16, N * sizeof(half));

  // Device memory allocation (bfloat16)
  __nv_bfloat16 *d_input_bf16, *d_output_bf16;
  cudaMalloc(&d_input_bf16, N * sizeof(__nv_bfloat16));
  cudaMalloc(&d_output_bf16, N * sizeof(__nv_bfloat16));

  // Copy input to device
  cudaMemcpy(d_input_f32, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_f16, h_input_f16, N * sizeof(half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_bf16, h_input_bf16, N * sizeof(__nv_bfloat16),
             cudaMemcpyHostToDevice);

  // Create CUDA events for timing
  cudaEvent_t start_f32, stop_f32, start_f16, stop_f16, start_bf16, stop_bf16;
  cudaEventCreate(&start_f32);
  cudaEventCreate(&stop_f32);
  cudaEventCreate(&start_f16);
  cudaEventCreate(&stop_f16);
  cudaEventCreate(&start_bf16);
  cudaEventCreate(&stop_bf16);

  // Warmup
  for (int i = 0; i < warmup; i++) {
    sigmod_f32<<<gridSize, blockSize>>>(d_input_f32, d_output_f32, N);
    sigmod_f16<<<gridSize, blockSize>>>(d_input_f16, d_output_f16, N);
    sigmod_bf16<<<gridSize, blockSize>>>(d_input_bf16, d_output_bf16, N);
  }
  cudaDeviceSynchronize();

  // Benchmark FP32
  cudaEventRecord(start_f32);
  for (int i = 0; i < iterations; i++) {
    sigmod_f32<<<gridSize, blockSize>>>(d_input_f32, d_output_f32, N);
  }
  cudaEventRecord(stop_f32);
  cudaEventSynchronize(stop_f32);

  // Benchmark FP16
  cudaEventRecord(start_f16);
  for (int i = 0; i < iterations; i++) {
    sigmod_f16<<<gridSize, blockSize>>>(d_input_f16, d_output_f16, N);
  }
  cudaEventRecord(stop_f16);
  cudaEventSynchronize(stop_f16);

  // Benchmark BF16
  cudaEventRecord(start_bf16);
  for (int i = 0; i < iterations; i++) {
    sigmod_bf16<<<gridSize, blockSize>>>(d_input_bf16, d_output_bf16, N);
  }
  cudaEventRecord(stop_bf16);
  cudaEventSynchronize(stop_bf16);

  // Calculate elapsed times
  float time_f32 = 0;
  float time_f16 = 0;
  float time_bf16 = 0;
  cudaEventElapsedTime(&time_f32, start_f32, stop_f32);
  cudaEventElapsedTime(&time_f16, start_f16, stop_f16);
  cudaEventElapsedTime(&time_bf16, start_bf16, stop_bf16);

  // Print performance results
  printf("\nPerformance Comparison (%d elements, %d iterations):\n", N,
         iterations);
  printf("------------------------------------------------------------\n");
  printf("FP32:  %.4f ms total, %.4f ms avg, %.2f GB/s\n", time_f32,
         time_f32 / iterations,
         (N * 2 * sizeof(float) * iterations) / (time_f32 / 1000.0) / 1e9);
  printf("FP16:  %.4f ms total, %.4f ms avg, %.2f GB/s (speedup: %.2fx)\n",
         time_f16, time_f16 / iterations,
         (N * 2 * sizeof(half) * iterations) / (time_f16 / 1000.0) / 1e9,
         time_f32 / time_f16);
  printf("BF16:  %.4f ms total, %.4f ms avg, %.2f GB/s (speedup: %.2fx)\n",
         time_bf16, time_bf16 / iterations,
         (N * 2 * sizeof(__nv_bfloat16) * iterations) / (time_bf16 / 1000.0) /
             1e9,
         time_f32 / time_bf16);

  // Copy output to host for accuracy check
  cudaMemcpy(h_output_f32, d_output_f32, N * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output_f16_half, d_output_f16, N * sizeof(half),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output_bf16_half, d_output_bf16, N * sizeof(__nv_bfloat16),
             cudaMemcpyDeviceToHost);

  // Convert half/bfloat16 output to float for comparison
  for (int i = 0; i < N; i++) {
    h_output_f16[i] = __half2float(h_output_f16_half[i]);
    h_output_bf16[i] = __bfloat162float(h_output_bf16_half[i]);
  }

  // Print first 10 results comparison
  printf("\nFirst 10 results comparison:\n");
  printf("%-10s %-12s %-12s %-12s %-12s %-12s\n", "Input", "FP32", "FP16",
         "Diff(F32-F16)", "BF16", "Diff(F32-BF16)");
  printf("---------------------------------------------------------------------"
         "-----\n");
  float max_diff_f16 = 0.0f;
  float avg_diff_f16 = 0.0f;
  float max_diff_bf16 = 0.0f;
  float avg_diff_bf16 = 0.0f;
  for (int i = 0; i < N; i++) {
    float diff_f16 = fabsf(h_output_f32[i] - h_output_f16[i]);
    float diff_bf16 = fabsf(h_output_f32[i] - h_output_bf16[i]);
    avg_diff_f16 += diff_f16;
    avg_diff_bf16 += diff_bf16;
    if (diff_f16 > max_diff_f16)
      max_diff_f16 = diff_f16;
    if (diff_bf16 > max_diff_bf16)
      max_diff_bf16 = diff_bf16;

    if (i < 10) {
      printf("%-10.4f %-12.6f %-12.6f %-12.8f %-12.6f %-12.8f\n", h_input[i],
             h_output_f32[i], h_output_f16[i], diff_f16, h_output_bf16[i],
             diff_bf16);
    }
  }
  avg_diff_f16 /= N;
  avg_diff_bf16 /= N;

  printf("\nAccuracy Statistics (vs FP32):\n");
  printf("------------------------------------------------------------\n");
  printf("FP16: Max diff = %.8f, Avg diff = %.8f\n", max_diff_f16,
         avg_diff_f16);
  printf("BF16: Max diff = %.8f, Avg diff = %.8f\n", max_diff_bf16,
         avg_diff_bf16);

  // Cleanup
  cudaEventDestroy(start_f32);
  cudaEventDestroy(stop_f32);
  cudaEventDestroy(start_f16);
  cudaEventDestroy(stop_f16);
  cudaEventDestroy(start_bf16);
  cudaEventDestroy(stop_bf16);

  delete[] h_input;
  delete[] h_output_f32;
  delete[] h_output_f16;
  delete[] h_output_bf16;
  delete[] h_input_f16;
  delete[] h_output_f16_half;
  delete[] h_input_bf16;
  delete[] h_output_bf16_half;
  cudaFree(d_input_f32);
  cudaFree(d_output_f32);
  cudaFree(d_input_f16);
  cudaFree(d_output_f16);
  cudaFree(d_input_bf16);
  cudaFree(d_output_bf16);

  return 0;
}