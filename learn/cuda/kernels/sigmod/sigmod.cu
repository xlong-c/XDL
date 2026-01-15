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

__global__ void sigmod_f32(float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = -fminf(fmaxf(input[idx], MIN_EXP_F32), MAX_EXP_F32);
    output[idx] = 1.0f / (1.0f + expf(x));
  }
}

int main() {
  const int N = 1024;
  const int mask = N;
  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;

  // Host memory allocation
  float *h_input = new float[N];
  float *h_output = new float[N];

  // Initialize input data
  for (int i = 0; i < N; i++) {
    h_input[i] = (float)i / N - 0.5f; // Range: -0.5 to 0.5
  }

  // Device memory allocation
  float *d_input, *d_output;
  cudaMalloc(&d_input, N * sizeof(float));
  cudaMalloc(&d_output, N * sizeof(float));

  // Copy input to device
  cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel (float32)
  sigmod_f32<<<gridSize, blockSize>>>(d_input, d_output, mask);

  // Copy output to host
  cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Print first 10 results
  printf("First 10 results:\n");
  for (int i = 0; i < 10; i++) {
    printf("sigmod(%.2f) = %.6f\n", h_input[i], h_output[i]);
  }

  // Cleanup
  delete[] h_input;
  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}