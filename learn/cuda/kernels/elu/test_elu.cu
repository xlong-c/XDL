#include <algorithm>
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

// 验证结果（简单版本）
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

// 详细的误差分析函数
void analyze_errors(float *cpu_result, float *gpu_result, int n, 
                    const char* kernel_name, float epsilon = 1e-5f);

// 综合对比函数
void compare_kernel_errors(float *cpu_result, float *gpu_scalar, float *gpu_vector, float *gpu_half, 
                           int n, const char* test_name);

// 详细的误差分析函数实现
void analyze_errors(float *cpu_result, float *gpu_result, int n, 
                    const char* kernel_name, float epsilon) {
  printf("\n=== 误差分析: %s ===\n", kernel_name);
  
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double mae = 0.0;  // 平均绝对误差
  double mse = 0.0;  // 均方误差
  int error_count = 0;
  int max_error_idx = 0;
  int max_rel_error_idx = 0;
  
  for (int i = 0; i < n; i++) {
    float cpu_val = cpu_result[i];
    float gpu_val = gpu_result[i];
    float abs_error = fabs(cpu_val - gpu_val);
    float rel_error = 0.0;
    
    if (fabs(cpu_val) > 1e-10f) {  // 避免除以零
      rel_error = abs_error / fabs(cpu_val);
    }
    
    mae += abs_error;
    mse += abs_error * abs_error;
    
    if (abs_error > max_abs_error) {
      max_abs_error = abs_error;
      max_error_idx = i;
    }
    
    if (rel_error > max_rel_error) {
      max_rel_error = rel_error;
      max_rel_error_idx = i;
    }
    
    if (abs_error > epsilon) {
      error_count++;
    }
  }
  
  mae /= n;
  mse /= n;
  double rmse = sqrt(mse);
  
  printf("样本数量: %d\n", n);
  printf("容差阈值: %.2e\n", epsilon);
  printf("超出容差的样本数: %d (%.4f%%)\n", error_count, (error_count * 100.0) / n);
  printf("最大绝对误差: %.6e (索引: %d)\n", max_abs_error, max_error_idx);
  printf("  对应值: CPU=%.6f, GPU=%.6f\n", cpu_result[max_error_idx], gpu_result[max_error_idx]);
  printf("最大相对误差: %.6e (索引: %d)\n", max_rel_error, max_rel_error_idx);
  printf("  对应值: CPU=%.6f, GPU=%.6f\n", cpu_result[max_rel_error_idx], gpu_result[max_rel_error_idx]);
  printf("平均绝对误差 (MAE): %.6e\n", mae);
  printf("均方根误差 (RMSE): %.6e\n", rmse);
  
  // 误差分布统计
  const int num_bins = 10;
  int bins[num_bins] = {0};
  double bin_limits[num_bins + 1] = {
    0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, INFINITY
  };
  
  for (int i = 0; i < n; i++) {
    float abs_error = fabs(cpu_result[i] - gpu_result[i]);
    for (int b = 0; b < num_bins; b++) {
      if (abs_error >= bin_limits[b] && abs_error < bin_limits[b + 1]) {
        bins[b]++;
        break;
      }
    }
  }
  
  printf("\n误差分布:\n");
  for (int b = 0; b < num_bins; b++) {
    printf("  [%.0e, %.0e): %d (%.2f%%)\n", 
           bin_limits[b], bin_limits[b + 1], 
           bins[b], (bins[b] * 100.0) / n);
  }
  
  // 特殊值分析
  int positive_count = 0;
  int negative_count = 0;
  int zero_count = 0;
  double positive_mae = 0.0;
  double negative_mae = 0.0;
  
  for (int i = 0; i < n; i++) {
    float cpu_val = cpu_result[i];
    float error = fabs(cpu_val - gpu_result[i]);
    
    if (cpu_val > 1e-10f) {
      positive_count++;
      positive_mae += error;
    } else if (cpu_val < -1e-10f) {
      negative_count++;
      negative_mae += error;
    } else {
      zero_count++;
    }
  }
  
  if (positive_count > 0) positive_mae /= positive_count;
  if (negative_count > 0) negative_mae /= negative_count;
  
  printf("\n按输入符号的误差分析:\n");
  printf("  正数输入: %d 个, 平均误差: %.6e\n", positive_count, positive_mae);
  printf("  负数输入: %d 个, 平均误差: %.6e\n", negative_count, negative_mae);
  printf("  零值输入: %d 个\n", zero_count);
  
  // 数值稳定性分析
  int large_error_count = 0;
  int moderate_error_count = 0;
  int small_error_count = 0;
  
  for (int i = 0; i < n; i++) {
    float abs_error = fabs(cpu_result[i] - gpu_result[i]);
    if (abs_error > 1e-3f) {
      large_error_count++;
    } else if (abs_error > 1e-6f) {
      moderate_error_count++;
    } else {
      small_error_count++;
    }
  }
  
  printf("\n数值稳定性:\n");
  printf("  小误差 (<1e-6): %d (%.2f%%)\n", small_error_count, (small_error_count * 100.0) / n);
  printf("  中误差 (1e-6~1e-3): %d (%.2f%%)\n", moderate_error_count, (moderate_error_count * 100.0) / n);
  printf("  大误差 (>1e-3): %d (%.2f%%)\n", large_error_count, (large_error_count * 100.0) / n);
  
  printf("=== 误差分析结束 ===\n\n");
}

// 测试函数
void test_elu_kernels(size_t N) {
  printf("Testing ELU kernels with array size: %zu elements\n", N);
  printf("Total memory (float): %.2f MB\n", (N * sizeof(float)) / (1024.0 * 1024.0));

  // 分配主机内存
  float *h_x = new float[N];
  float *h_y_cpu = new float[N];
  float *h_y_gpu = new float[N];
  float *h_y_gpu_scalar = new float[N];  // 用于标量版本结果
  float *h_y_gpu_vector = new float[N];  // 用于向量化版本结果

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
    
    // 复制标量版本结果进行分析
    err = cudaMemcpy(h_y_gpu_scalar, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
      int check_count = std::min(N, (size_t)10000);
      printf("\nVerifying scalar results...\n");
      if (verify_results(h_y_cpu, h_y_gpu_scalar, check_count)) {
        printf("✓ Scalar results match! (checked first %d elements)\n", check_count);
      } else {
        printf("✗ Scalar results do not match!\n");
      }
      analyze_errors(h_y_cpu, h_y_gpu_scalar, check_count, "elu_f32_scalar_kernel");
    }
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
  err = cudaMemcpy(h_y_gpu_vector, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
  } else {
    printf("Verifying float4 results...\n");
    int check_count = std::min(N, (size_t)10000);
    if (verify_results(h_y_cpu, h_y_gpu_vector, check_count)) {
      printf("✓ Results match! (checked first %d elements)\n", check_count);
    } else {
      printf("✗ Results do not match!\n");
    }
    // 进行详细的误差分析
    analyze_errors(h_y_cpu, h_y_gpu_vector, check_count, "elu_f32x4_kernel (float4)");
  }

  // 4. 测试 elu_fp16x2_kernel（半精度版本）
  printf("\n4. Testing elu_fp16x2_kernel (half precision)...\n");

  // 分配half精度内存
  half *h_x_half = new half[N];
  half *h_y_gpu_half = new half[N];
  float *h_y_gpu_half_float = new float[N];  // 用于存储转换后的half结果
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
            
            // 转换half结果为float用于误差分析
            for (int i = 0; i < N; i++) {
              h_y_gpu_half_float[i] = __half2float(h_y_gpu_half[i]);
            }
            
            for (int i = 0; i < check_count; i++) {
              float diff = fabs(h_y_cpu[i] - h_y_gpu_half_float[i]);
              if (diff > 1e-3f) {         // half精度较低，放宽容差
                if (mismatch_count < 3) { // 只打印前3个不匹配
                  printf("  Mismatch at index %d: CPU=%f, GPU(half)=%f (diff=%f)\n", 
                         i, h_y_cpu[i], h_y_gpu_half_float[i], diff);
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
            
            // 进行详细的误差分析（使用更宽松的容差）
            analyze_errors(h_y_cpu, h_y_gpu_half_float, check_count, "elu_fp16x2_kernel (half2)", 1e-3f);
          }
        }

        // 清理half精度设备内存
        cudaFree(d_x_half);
        cudaFree(d_y_half);
      }
    }
  }

  // 进行综合误差对比
  int compare_count = std::min(N, (size_t)10000);
  char size_name[32];
  if (N >= 1024 * 1024 * 128) {
    snprintf(size_name, sizeof(size_name), "128M elements");
  } else if (N >= 1024 * 1024 * 16) {
    snprintf(size_name, sizeof(size_name), "16M elements");
  } else {
    snprintf(size_name, sizeof(size_name), "1M elements");
  }
  
  compare_kernel_errors(h_y_cpu, h_y_gpu_scalar, h_y_gpu_vector, h_y_gpu_half_float, 
                       compare_count, size_name);

  // 清理
  cudaFree(d_x);
  cudaFree(d_y);
  delete[] h_x;
  delete[] h_y_cpu;
  delete[] h_y_gpu;
  delete[] h_x_half;
  delete[] h_y_gpu_half;
  delete[] h_y_gpu_scalar;
  delete[] h_y_gpu_vector;
  delete[] h_y_gpu_half_float;

  printf("\nTest completed!\n");
}

// 综合对比函数
void compare_kernel_errors(float *cpu_result, float *gpu_scalar, float *gpu_vector, float *gpu_half, 
                           int n, const char* test_name) {
  printf("\n==========================================\n");
  printf("综合误差对比: %s\n", test_name);
  printf("==========================================\n");
  
  // 计算各内核的MAE
  double mae_scalar = 0.0;
  double mae_vector = 0.0;
  double mae_half = 0.0;
  
  for (int i = 0; i < n; i++) {
    mae_scalar += fabs(cpu_result[i] - gpu_scalar[i]);
    mae_vector += fabs(cpu_result[i] - gpu_vector[i]);
    mae_half += fabs(cpu_result[i] - gpu_half[i]);
  }
  
  mae_scalar /= n;
  mae_vector /= n;
  mae_half /= n;
  
  printf("平均绝对误差 (MAE) 对比:\n");
  printf("  标量版本: %.6e\n", mae_scalar);
  printf("  向量化版本: %.6e\n", mae_vector);
  printf("  半精度版本: %.6e\n", mae_half);
  printf("  向量化 vs 标量: %.2fx\n", mae_scalar / mae_vector);
  printf("  半精度 vs 标量: %.2fx\n", mae_scalar / mae_half);
  
  // 计算误差比率
  int better_vector = 0;
  int better_half = 0;
  int equal_vector = 0;
  int equal_half = 0;
  
  for (int i = 0; i < n; i++) {
    float error_scalar = fabs(cpu_result[i] - gpu_scalar[i]);
    float error_vector = fabs(cpu_result[i] - gpu_vector[i]);
    float error_half = fabs(cpu_result[i] - gpu_half[i]);
    
    if (error_vector < error_scalar - 1e-10f) better_vector++;
    else if (fabs(error_vector - error_scalar) < 1e-10f) equal_vector++;
    
    if (error_half < error_scalar - 1e-10f) better_half++;
    else if (fabs(error_half - error_scalar) < 1e-10f) equal_half++;
  }
  
  printf("\n误差改进统计:\n");
  printf("  向量化版本比标量版本误差更小: %d (%.2f%%)\n", 
         better_vector, (better_vector * 100.0) / n);
  printf("  向量化版本与标量版本误差相等: %d (%.2f%%)\n", 
         equal_vector, (equal_vector * 100.0) / n);
  printf("  半精度版本比标量版本误差更小: %d (%.2f%%)\n", 
         better_half, (better_half * 100.0) / n);
  printf("  半精度版本与标量版本误差相等: %d (%.2f%%)\n", 
         equal_half, (equal_half * 100.0) / n);
  
  // 数值范围分析
  const int num_ranges = 5;
  const char* range_names[] = {"[-2.0, -1.0)", "[-1.0, -0.1)", "[-0.1, 0.1)", "[0.1, 1.0)", "[1.0, 2.0]"};
  double range_limits[] = {-2.0f, -1.0f, -0.1f, 0.1f, 1.0f, 2.0f};
  
  double mae_scalar_by_range[num_ranges] = {0};
  double mae_vector_by_range[num_ranges] = {0};
  double mae_half_by_range[num_ranges] = {0};
  int counts_by_range[num_ranges] = {0};
  
  for (int i = 0; i < n; i++) {
    float x = cpu_result[i]; // 注意：这里应该是输入值，但我们需要原始输入
    // 由于我们只有输出值，这里简化处理
    // 在实际应用中，应该传入原始输入值
    
    for (int r = 0; r < num_ranges; r++) {
      if (x >= range_limits[r] && x < range_limits[r + 1]) {
        mae_scalar_by_range[r] += fabs(cpu_result[i] - gpu_scalar[i]);
        mae_vector_by_range[r] += fabs(cpu_result[i] - gpu_vector[i]);
        mae_half_by_range[r] += fabs(cpu_result[i] - gpu_half[i]);
        counts_by_range[r]++;
        break;
      }
    }
  }
  
  printf("\n按输出值范围的误差分析:\n");
  for (int r = 0; r < num_ranges; r++) {
    if (counts_by_range[r] > 0) {
      mae_scalar_by_range[r] /= counts_by_range[r];
      mae_vector_by_range[r] /= counts_by_range[r];
      mae_half_by_range[r] /= counts_by_range[r];
      
      printf("  范围 %s (%d 个样本):\n", range_names[r], counts_by_range[r]);
      printf("    标量: %.6e, 向量化: %.6e, 半精度: %.6e\n",
             mae_scalar_by_range[r], mae_vector_by_range[r], mae_half_by_range[r]);
    }
  }
  
  printf("==========================================\n\n");
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