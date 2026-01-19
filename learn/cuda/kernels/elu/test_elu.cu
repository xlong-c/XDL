#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 包含ELU内核
#include "elu.cu"

// CPU参考实现
void elu_cpu_reference(float *x, float *y, int n) {
  const float ALPHA_VALUE = 1.0f;
  for (int i = 0; i < n; i++) {
    y[i] = x[i] > 0.f ? x[i] : ALPHA_VALUE * (expf(x[i]) - 1.f);
  }
}

// 验证结果
bool verify_results(float *cpu_result, float *gpu_result, int n,
                    float epsilon = 1e-5f) {
  for (int i = 0; i < n; i++) {
    if (fabs(cpu_result[i] - gpu_result[i]) > epsilon) {
      printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, cpu_result[i], gpu_result[i]);
      return false;
    }
  }
  return true;
}

// 测试函数
void test_elu_kernels(size_t N) {
  printf("Testing ELU kernels with array size: %zu elements\n", N);
  printf("Total memory (float): %.2f MB\n", (N * sizeof(float)) / (1024.0 * 1024.0));

  // 分配主机内存
  float *h_x = new float[N];
  float *h_y_cpu = new float[N];
  float *h_y_gpu = new float[N];

  // 初始化随机数据
  srand(time(NULL));
  for (size_t i = 0; i < N; i++) {
    h_x[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f; // 范围 [-2.0, 2.0]
  }

  // 1. 运行CPU参考实现
  printf("\n1. Running CPU reference implementation...\n");
  clock_t start_cpu = clock();
  elu_cpu_reference(h_x, h_y_cpu, N);
  clock_t end_cpu = clock();
  double cpu_duration = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU time: %.6f seconds\n", cpu_duration);

  // 分配设备内存
  float *d_x = nullptr;
  float *d_y = nullptr;
  cudaError_t err;

  err = cudaMalloc(&d_x, N * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for d_x: %s\n", cudaGetErrorString(err));
    delete[] h_x;
    delete[] h_y_cpu;
    delete[] h_y_gpu;
    return;
  }

  err = cudaMalloc(&d_y, N * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for d_y: %s\n", cudaGetErrorString(err));
    cudaFree(d_x);
    delete[] h_x;
    delete[] h_y_cpu;
    delete[] h_y_gpu;
    return;
  }

  // 复制数据到设备
  err = cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y_cpu;
    delete[] h_y_gpu;
    return;
  }

  const int iterations = 10;
  const int block_size = 256;

  // 2. 测试 elu_f32_scalar_kernel（标量版本）
  printf("\n2. Testing elu_f32_scalar_kernel (baseline)...\n");
  const int grid_size_scalar = (N + block_size - 1) / block_size;

  printf("Grid size: %d, Block size: %d\n", grid_size_scalar, block_size);
  printf("Total threads: %d\n", grid_size_scalar * block_size);

  // 预热
  elu_f32_scalar_kernel<<<grid_size_scalar, block_size>>>(d_x, d_y, N);
  cudaDeviceSynchronize();

  // 计时运行
  clock_t start_gpu = clock();
  for (int i = 0; i < iterations; i++) {
    elu_f32_scalar_kernel<<<grid_size_scalar, block_size>>>(d_x, d_y, N);
  }
  cudaDeviceSynchronize();
  clock_t end_gpu = clock();
  double gpu_duration_scalar = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC / iterations;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel error (scalar): %s\n", cudaGetErrorString(err));
  } else {
    printf("GPU time (scalar): %.6f seconds\n", gpu_duration_scalar);
    printf("Speedup vs CPU: %.2fx\n", cpu_duration / gpu_duration_scalar);

    // 计算内存带宽
    double total_bytes_scalar = 2.0 * N * sizeof(float);
    double bandwidth_gb_s_scalar =
        (total_bytes_scalar / (1024.0 * 1024.0 * 1024.0)) /
        gpu_duration_scalar;
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth_gb_s_scalar);
  }

  // 3. 测试 elu_f32x4_kernel（向量化版本）
  printf("\n3. Testing elu_f32x4_kernel (vectorized float4)...\n");
  const int grid_size_vector = (N + block_size * 4 - 1) / (block_size * 4);

  printf("Grid size: %d, Block size: %d\n", grid_size_vector, block_size);
  printf("Total threads: %d\n", grid_size_vector * block_size);

  // 预热
  elu_f32x4_kernel<<<grid_size_vector, block_size>>>(d_x, d_y, N);
  cudaDeviceSynchronize();

  // 计时运行
  start_gpu = clock();
  for (int i = 0; i < iterations; i++) {
    elu_f32x4_kernel<<<grid_size_vector, block_size>>>(d_x, d_y, N);
  }
  cudaDeviceSynchronize();
  end_gpu = clock();
  double gpu_duration_vector = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC / iterations;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel error (vectorized): %s\n", cudaGetErrorString(err));
  } else {
    printf("GPU time (float4): %.6f seconds\n", gpu_duration_vector);
    printf("Speedup vs CPU: %.2fx\n", cpu_duration / gpu_duration_vector);
    printf("Speedup vs scalar: %.2fx\n", gpu_duration_scalar / gpu_duration_vector);

    // 计算内存带宽
    double total_bytes_vector = 2.0 * N * sizeof(float);
    double bandwidth_gb_s_vector =
        (total_bytes_vector / (1024.0 * 1024.0 * 1024.0)) /
        gpu_duration_vector;
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth_gb_s_vector);
  }

  // 复制结果回主机并验证
  err = cudaMemcpy(h_y_gpu, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
  } else {
    printf("Verifying float4 results...\n");
    if (verify_results(h_y_cpu, h_y_gpu, std::min(N, (size_t)10000))) {
      printf("✓ Results match! (checked first 10000 elements)\n");
    } else {
      printf("✗ Results do not match!\n");
    }
  }

  // 4. 测试 elu_fp16x2_kernel（半精度版本）
  printf("\n4. Testing elu_fp16x2_kernel (half precision)...\n");

  // 分配half精度内存
  half *h_x_half = new half[N];
  half *h_y_gpu_half = new half[N];
  half *d_x_half = nullptr;
  half *d_y_half = nullptr;

  // 将float数据转换为half
  for (size_t i = 0; i < N; i++) {
    h_x_half[i] = __float2half(h_x[i]);
  }

  // 分配设备内存
  err = cudaMalloc(&d_x_half, N * sizeof(half));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed for d_x_half: %s\n", cudaGetErrorString(err));
  } else {
    err = cudaMalloc(&d_y_half, N * sizeof(half));
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed for d_y_half: %s\n", cudaGetErrorString(err));
      cudaFree(d_x_half);
    } else {
      // 复制数据到设备
      err = cudaMemcpy(d_x_half, h_x_half, N * sizeof(half),
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for half data: %s\n", cudaGetErrorString(err));
      } else {
        // 设置CUDA内核参数
        const int grid_size_half = (N + block_size * 2 - 1) / (block_size * 2);

        printf("Grid size: %d, Block size: %d\n", grid_size_half, block_size);
        printf("Total threads: %d\n", grid_size_half * block_size);

        // 预热
        elu_fp16x2_kernel<<<grid_size_half, block_size>>>(d_x_half, d_y_half,
                                                          N);
        cudaDeviceSynchronize();

        // 计时运行
        clock_t start_gpu_half = clock();
        for (int i = 0; i < iterations; i++) {
          elu_fp16x2_kernel<<<grid_size_half, block_size>>>(d_x_half, d_y_half,
                                                            N);
        }
        cudaDeviceSynchronize();
        clock_t end_gpu_half = clock();
        double gpu_duration_half = (double)(end_gpu_half - start_gpu_half) / CLOCKS_PER_SEC / iterations;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
          fprintf(stderr, "CUDA kernel error (half): %s\n", cudaGetErrorString(err));
        } else {
          printf("GPU time (half2): %.6f seconds\n", gpu_duration_half);
          printf("Speedup vs CPU: %.2fx\n", cpu_duration / gpu_duration_half);
          printf("Speedup vs scalar: %.2fx\n", gpu_duration_scalar / gpu_duration_half);

          // 计算内存带宽
          double total_bytes_half = 2.0 * N * sizeof(half);
          double bandwidth_gb_s_half =
              (total_bytes_half / (1024.0 * 1024.0 * 1024.0)) /
              gpu_duration_half;
          printf("Memory bandwidth: %.2f GB/s\n", bandwidth_gb_s_half);

          // 复制结果回主机
          err = cudaMemcpy(h_y_gpu_half, d_y_half, N * sizeof(half),
                           cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed for half results: %s\n", cudaGetErrorString(err));
          } else {
            // 验证half精度结果
            printf("Verifying half precision results...\n");
            bool half_match = true;
            int check_count = std::min(N, (size_t)10000);
            int mismatch_count = 0;
            for (int i = 0; i < check_count; i++) {
              float gpu_val = __half2float(h_y_gpu_half[i]);
              float diff = fabs(h_y_cpu[i] - gpu_val);
              if (diff > 1e-3f) {         // half精度较低，放宽容差
                if (mismatch_count < 3) { // 只打印前3个不匹配
                  printf("  Mismatch at index %d: CPU=%f, GPU(half)=%f (diff=%f)\n", 
                         i, h_y_cpu[i], gpu_val, diff);
                }
                mismatch_count++;
                half_match = false;
              }
            }
            if (half_match) {
              printf("✓ Half precision results match within tolerance!\n");
            } else {
              printf("✗ Half precision: %d mismatches out of %d checked elements\n", 
                     mismatch_count, check_count);
              printf("  (expected due to lower precision of half)\n");
            }
          }
        }

        // 清理half精度设备内存
        cudaFree(d_x_half);
        cudaFree(d_y_half);
      }
    }
  }

  // 清理
  cudaFree(d_x);
  cudaFree(d_y);
  delete[] h_x;
  delete[] h_y_cpu;
  delete[] h_y_gpu;
  delete[] h_x_half;
  delete[] h_y_gpu_half;

  printf("\nTest completed!\n");
}

int main() {
  // 测试不同大小的数组
  const size_t sizes[] = {
      1024 * 1024,      // 1M elements
      1024 * 1024 * 16, // 16M elements
      1024 * 1024 * 128 // 128M elements (原始要求)
  };

  const char *size_names[] = {"1M", "16M", "128M"};

  for (int i = 0; i < 3; i++) {
    printf("\n==========================================\n");
    printf("Testing with %s elements\n", size_names[i]);
    printf("==========================================\n");
    test_elu_kernels(sizes[i]);
  }

  return 0;
}