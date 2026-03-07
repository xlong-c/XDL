#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t error_code__ = (call);                                        \
    if (error_code__ != cudaSuccess) {                                        \
      std::fprintf(stderr, "CUDA error: %s @ %s:%d\n",                      \
                   cudaGetErrorString(error_code__), __FILE__, __LINE__);     \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

struct BenchmarkResult {
  const char *name = "";
  const char *teaching_point = "";
  float max_abs_error = 0.0f;
  float avg_time_ms = 0.0f;
  double tflops = 0.0;
  bool passed = false;
};

int parse_positive_int(const char *text, const char *name) {
  char *end_ptr = nullptr;
  const long value = std::strtol(text, &end_ptr, 10);
  if (end_ptr == text || *end_ptr != '\0' || value <= 0 || value > 1'000'000) {
    std::fprintf(stderr, "非法 %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

void print_usage(const char *program_name) {
  std::printf("用法:\n");
  std::printf("  %s\n", program_name);
  std::printf("  %s M N K\n\n", program_name);
  std::printf("默认尺寸: M=N=K=512\n");
  std::printf("建议第一次先直接运行默认尺寸, 再尝试改成 1024 观察不同版本的性能变化。\n");
}

void fill_matrix(std::vector<float> &matrix, const int rows, const int cols) {
  for (int row_index = 0; row_index < rows; ++row_index) {
    for (int col_index = 0; col_index < cols; ++col_index) {
      const int linear_index = row_index * cols + col_index;
      const float base_value = static_cast<float>((linear_index % 17) - 8) * 0.125f;
      const float row_bias = static_cast<float>((row_index % 5) - 2) * 0.01f;
      matrix[linear_index] = base_value + row_bias;
    }
  }
}

void cpu_sgemm_reference(const std::vector<float> &host_a,
                         const std::vector<float> &host_b,
                         std::vector<float> &host_c,
                         const int rows_m,
                         const int cols_n,
                         const int depth_k) {
  for (int row_index = 0; row_index < rows_m; ++row_index) {
    for (int col_index = 0; col_index < cols_n; ++col_index) {
      float sum_value = 0.0f;
      for (int k_index = 0; k_index < depth_k; ++k_index) {
        sum_value += host_a[row_index * depth_k + k_index] *
                     host_b[k_index * cols_n + col_index];
      }
      host_c[row_index * cols_n + col_index] = sum_value;
    }
  }
}

__global__ void sgemm_step0_naive_kernel(const float *matrix_a,
                                         const float *matrix_b,
                                         float *matrix_c,
                                         const int rows_m,
                                         const int cols_n,
                                         const int depth_k) {
  const int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_index >= rows_m || col_index >= cols_n) {
    return;
  }

  float sum_value = 0.0f;
  for (int k_index = 0; k_index < depth_k; ++k_index) {
    sum_value += matrix_a[row_index * depth_k + k_index] *
                 matrix_b[k_index * cols_n + col_index];
  }

  matrix_c[row_index * cols_n + col_index] = sum_value;
}

/*
 * Step 1:
 * - 第一次引入 shared memory
 * - block 负责一个 16x16 的输出 tile
 * - 每个线程仍然只计算 1 个输出元素
 */
template <int BlockRowsM = 16, int BlockColsN = 16, int BlockDepthK = 8>
__global__ void sgemm_step1_shared_tiled_kernel(const float *matrix_a,
                                                const float *matrix_b,
                                                float *matrix_c,
                                                const int rows_m,
                                                const int cols_n,
                                                const int depth_k) {
  __shared__ float shared_a[BlockRowsM][BlockDepthK];
  __shared__ float shared_b[BlockDepthK][BlockColsN];

  const int thread_row = threadIdx.y;
  const int thread_col = threadIdx.x;
  const int thread_index = thread_row * blockDim.x + thread_col;
  const int threads_per_block = blockDim.x * blockDim.y;

  const int global_row = blockIdx.y * BlockRowsM + thread_row;
  const int global_col = blockIdx.x * BlockColsN + thread_col;

  float sum_value = 0.0f;

  for (int block_k_start = 0; block_k_start < depth_k; block_k_start += BlockDepthK) {
    for (int linear_index = thread_index;
         linear_index < BlockRowsM * BlockDepthK;
         linear_index += threads_per_block) {
      const int tile_row = linear_index / BlockDepthK;
      const int tile_col = linear_index % BlockDepthK;
      const int source_row = blockIdx.y * BlockRowsM + tile_row;
      const int source_col = block_k_start + tile_col;
      shared_a[tile_row][tile_col] =
          (source_row < rows_m && source_col < depth_k)
              ? matrix_a[source_row * depth_k + source_col]
              : 0.0f;
    }

    for (int linear_index = thread_index;
         linear_index < BlockDepthK * BlockColsN;
         linear_index += threads_per_block) {
      const int tile_row = linear_index / BlockColsN;
      const int tile_col = linear_index % BlockColsN;
      const int source_row = block_k_start + tile_row;
      const int source_col = blockIdx.x * BlockColsN + tile_col;
      shared_b[tile_row][tile_col] =
          (source_row < depth_k && source_col < cols_n)
              ? matrix_b[source_row * cols_n + source_col]
              : 0.0f;
    }

    __syncthreads();

    for (int k_index = 0; k_index < BlockDepthK; ++k_index) {
      sum_value += shared_a[thread_row][k_index] * shared_b[k_index][thread_col];
    }

    __syncthreads();
  }

  if (global_row < rows_m && global_col < cols_n) {
    matrix_c[global_row * cols_n + global_col] = sum_value;
  }
}

/*
 * Step 2:
 * - 第一次引入 register blocking
 * - 每个线程计算 TM x 1 输出块
 * - 目的是让一个线程复用同一列上的 B 数据
 */
template <int BlockRowsM = 32,
          int BlockColsN = 32,
          int BlockDepthK = 8,
          int ThreadRowsM = 4>
__global__ void sgemm_step2_thread_tile_1d_kernel(const float *matrix_a,
                                                  const float *matrix_b,
                                                  float *matrix_c,
                                                  const int rows_m,
                                                  const int cols_n,
                                                  const int depth_k) {
  __shared__ float shared_a[BlockRowsM][BlockDepthK];
  __shared__ float shared_b[BlockDepthK][BlockColsN];

  const int thread_row_group = threadIdx.y;
  const int thread_col = threadIdx.x;
  const int thread_index = thread_row_group * blockDim.x + thread_col;
  const int threads_per_block = blockDim.x * blockDim.y;

  const int block_row_start = blockIdx.y * BlockRowsM;
  const int block_col_start = blockIdx.x * BlockColsN;
  const int global_col = block_col_start + thread_col;

  float accumulators[ThreadRowsM] = {0.0f};

  for (int block_k_start = 0; block_k_start < depth_k; block_k_start += BlockDepthK) {
    for (int linear_index = thread_index;
         linear_index < BlockRowsM * BlockDepthK;
         linear_index += threads_per_block) {
      const int tile_row = linear_index / BlockDepthK;
      const int tile_col = linear_index % BlockDepthK;
      const int source_row = block_row_start + tile_row;
      const int source_col = block_k_start + tile_col;
      shared_a[tile_row][tile_col] =
          (source_row < rows_m && source_col < depth_k)
              ? matrix_a[source_row * depth_k + source_col]
              : 0.0f;
    }

    for (int linear_index = thread_index;
         linear_index < BlockDepthK * BlockColsN;
         linear_index += threads_per_block) {
      const int tile_row = linear_index / BlockColsN;
      const int tile_col = linear_index % BlockColsN;
      const int source_row = block_k_start + tile_row;
      const int source_col = block_col_start + tile_col;
      shared_b[tile_row][tile_col] =
          (source_row < depth_k && source_col < cols_n)
              ? matrix_b[source_row * cols_n + source_col]
              : 0.0f;
    }

    __syncthreads();

    for (int k_index = 0; k_index < BlockDepthK; ++k_index) {
      const float b_value = shared_b[k_index][thread_col];
      #pragma unroll
      for (int row_offset = 0; row_offset < ThreadRowsM; ++row_offset) {
        const int shared_row = thread_row_group * ThreadRowsM + row_offset;
        accumulators[row_offset] += shared_a[shared_row][k_index] * b_value;
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int row_offset = 0; row_offset < ThreadRowsM; ++row_offset) {
    const int global_row = block_row_start + thread_row_group * ThreadRowsM + row_offset;
    if (global_row < rows_m && global_col < cols_n) {
      matrix_c[global_row * cols_n + global_col] = accumulators[row_offset];
    }
  }
}

/*
 * Step 3:
 * - 每个线程计算 TM x TN 输出块
 * - shared memory -> register -> outer-product 更新这条链条会更清晰
 */
template <int BlockRowsM = 64,
          int BlockColsN = 64,
          int BlockDepthK = 8,
          int ThreadRowsM = 4,
          int ThreadColsN = 4>
__global__ void sgemm_step3_thread_tile_2d_kernel(const float *matrix_a,
                                                  const float *matrix_b,
                                                  float *matrix_c,
                                                  const int rows_m,
                                                  const int cols_n,
                                                  const int depth_k) {
  __shared__ float shared_a[BlockRowsM][BlockDepthK];
  __shared__ float shared_b[BlockDepthK][BlockColsN];

  const int thread_tile_row = threadIdx.y;
  const int thread_tile_col = threadIdx.x;
  const int thread_index = thread_tile_row * blockDim.x + thread_tile_col;
  const int threads_per_block = blockDim.x * blockDim.y;

  const int block_row_start = blockIdx.y * BlockRowsM;
  const int block_col_start = blockIdx.x * BlockColsN;

  float accumulators[ThreadRowsM][ThreadColsN] = {0.0f};

  for (int block_k_start = 0; block_k_start < depth_k; block_k_start += BlockDepthK) {
    for (int linear_index = thread_index;
         linear_index < BlockRowsM * BlockDepthK;
         linear_index += threads_per_block) {
      const int tile_row = linear_index / BlockDepthK;
      const int tile_col = linear_index % BlockDepthK;
      const int source_row = block_row_start + tile_row;
      const int source_col = block_k_start + tile_col;
      shared_a[tile_row][tile_col] =
          (source_row < rows_m && source_col < depth_k)
              ? matrix_a[source_row * depth_k + source_col]
              : 0.0f;
    }

    for (int linear_index = thread_index;
         linear_index < BlockDepthK * BlockColsN;
         linear_index += threads_per_block) {
      const int tile_row = linear_index / BlockColsN;
      const int tile_col = linear_index % BlockColsN;
      const int source_row = block_k_start + tile_row;
      const int source_col = block_col_start + tile_col;
      shared_b[tile_row][tile_col] =
          (source_row < depth_k && source_col < cols_n)
              ? matrix_b[source_row * cols_n + source_col]
              : 0.0f;
    }

    __syncthreads();

    for (int k_index = 0; k_index < BlockDepthK; ++k_index) {
      float fragment_a[ThreadRowsM];
      float fragment_b[ThreadColsN];

      #pragma unroll
      for (int row_offset = 0; row_offset < ThreadRowsM; ++row_offset) {
        const int shared_row = thread_tile_row * ThreadRowsM + row_offset;
        fragment_a[row_offset] = shared_a[shared_row][k_index];
      }

      #pragma unroll
      for (int col_offset = 0; col_offset < ThreadColsN; ++col_offset) {
        const int shared_col = thread_tile_col * ThreadColsN + col_offset;
        fragment_b[col_offset] = shared_b[k_index][shared_col];
      }

      #pragma unroll
      for (int row_offset = 0; row_offset < ThreadRowsM; ++row_offset) {
        #pragma unroll
        for (int col_offset = 0; col_offset < ThreadColsN; ++col_offset) {
          accumulators[row_offset][col_offset] +=
              fragment_a[row_offset] * fragment_b[col_offset];
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int row_offset = 0; row_offset < ThreadRowsM; ++row_offset) {
    const int global_row = block_row_start + thread_tile_row * ThreadRowsM + row_offset;
    #pragma unroll
    for (int col_offset = 0; col_offset < ThreadColsN; ++col_offset) {
      const int global_col = block_col_start + thread_tile_col * ThreadColsN + col_offset;
      if (global_row < rows_m && global_col < cols_n) {
        matrix_c[global_row * cols_n + global_col] = accumulators[row_offset][col_offset];
      }
    }
  }
}

double compute_tflops(const int rows_m,
                      const int cols_n,
                      const int depth_k,
                      const float avg_time_ms) {
  const double total_flops = 2.0 * static_cast<double>(rows_m) *
                             static_cast<double>(cols_n) *
                             static_cast<double>(depth_k);
  return total_flops / (static_cast<double>(avg_time_ms) * 1.0e9);
}

template <typename LaunchFunc>
BenchmarkResult benchmark_kernel(const char *kernel_name,
                                 const char *teaching_point,
                                 LaunchFunc launch_kernel,
                                 float *device_c,
                                 std::vector<float> &host_c,
                                 const std::vector<float> &host_reference,
                                 const size_t bytes_c,
                                 const int rows_m,
                                 const int cols_n,
                                 const int depth_k,
                                 const float tolerance,
                                 const int warmup_iters,
                                 const int repeat_iters) {
  BenchmarkResult result;
  result.name = kernel_name;
  result.teaching_point = teaching_point;

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&stop_event));

  CHECK_CUDA(cudaMemset(device_c, 0, bytes_c));
  for (int iter_index = 0; iter_index < warmup_iters; ++iter_index) {
    launch_kernel();
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start_event));
  for (int iter_index = 0; iter_index < repeat_iters; ++iter_index) {
    launch_kernel();
  }
  CHECK_CUDA(cudaEventRecord(stop_event));
  CHECK_CUDA(cudaEventSynchronize(stop_event));
  CHECK_CUDA(cudaGetLastError());

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
  result.avg_time_ms = elapsed_ms / static_cast<float>(repeat_iters);
  result.tflops = compute_tflops(rows_m, cols_n, depth_k, result.avg_time_ms);

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes_c, cudaMemcpyDeviceToHost));
  for (size_t value_index = 0; value_index < host_c.size(); ++value_index) {
    result.max_abs_error = std::max(result.max_abs_error,
                                    std::fabs(host_c[value_index] - host_reference[value_index]));
  }
  result.passed = result.max_abs_error <= tolerance;

  CHECK_CUDA(cudaEventDestroy(stop_event));
  CHECK_CUDA(cudaEventDestroy(start_event));
  return result;
}

int main(int argc, char **argv) {
  constexpr int kDefaultSize = 512;
  constexpr int kWarmupIters = 3;
  constexpr int kRepeatIters = 10;
  constexpr float kTolerance = 1e-3f;

  int rows_m = kDefaultSize;
  int cols_n = kDefaultSize;
  int depth_k = kDefaultSize;

  if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                    std::strcmp(argv[1], "-h") == 0)) {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  if (argc == 4) {
    rows_m = parse_positive_int(argv[1], "M");
    cols_n = parse_positive_int(argv[2], "N");
    depth_k = parse_positive_int(argv[3], "K");
  } else if (argc != 1) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::fprintf(stderr, "未检测到 CUDA 设备。\n");
    return EXIT_FAILURE;
  }

  const size_t size_a = static_cast<size_t>(rows_m) * depth_k;
  const size_t size_b = static_cast<size_t>(depth_k) * cols_n;
  const size_t size_c = static_cast<size_t>(rows_m) * cols_n;
  const size_t bytes_a = size_a * sizeof(float);
  const size_t bytes_b = size_b * sizeof(float);
  const size_t bytes_c = size_c * sizeof(float);

  std::vector<float> host_a(size_a, 0.0f);
  std::vector<float> host_b(size_b, 0.0f);
  std::vector<float> host_c(size_c, 0.0f);
  std::vector<float> host_reference(size_c, 0.0f);

  fill_matrix(host_a, rows_m, depth_k);
  fill_matrix(host_b, depth_k, cols_n);
  cpu_sgemm_reference(host_a, host_b, host_reference, rows_m, cols_n, depth_k);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a, bytes_a));
  CHECK_CUDA(cudaMalloc(&device_b, bytes_b));
  CHECK_CUDA(cudaMalloc(&device_c, bytes_c));
  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes_b, cudaMemcpyHostToDevice));

  const dim3 step0_block(16, 16);
  const dim3 step0_grid((cols_n + step0_block.x - 1) / step0_block.x,
                        (rows_m + step0_block.y - 1) / step0_block.y);

  const dim3 step1_block(16, 16);
  const dim3 step1_grid((cols_n + 16 - 1) / 16,
                        (rows_m + 16 - 1) / 16);

  const dim3 step2_block(32, 8);
  const dim3 step2_grid((cols_n + 32 - 1) / 32,
                        (rows_m + 32 - 1) / 32);

  const dim3 step3_block(16, 16);
  const dim3 step3_grid((cols_n + 64 - 1) / 64,
                        (rows_m + 64 - 1) / 64);

  std::printf("教学版 step-by-step SGEMM\n");
  std::printf("矩阵尺寸: M=%d, N=%d, K=%d\n", rows_m, cols_n, depth_k);
  std::printf("你应该按下面顺序理解这些 kernel:\n");
  std::printf("  step0: naïve, 每线程 1 个输出, 直接从 global memory 读\n");
  std::printf("  step1: block tiling + shared memory\n");
  std::printf("  step2: thread tiling(1D), 每线程负责 TM x 1\n");
  std::printf("  step3: thread tiling(2D), 每线程负责 TM x TN\n\n");

  std::vector<BenchmarkResult> results;
  results.push_back(benchmark_kernel(
      "step0_naive",
      "先写对: 一个线程只算一个输出元素",
      [&]() {
        sgemm_step0_naive_kernel<<<step0_grid, step0_block>>>(device_a,
                                                              device_b,
                                                              device_c,
                                                              rows_m,
                                                              cols_n,
                                                              depth_k);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTolerance,
      kWarmupIters,
      kRepeatIters));

  results.push_back(benchmark_kernel(
      "step1_shared_tiled",
      "第一次显式利用 block 级数据复用",
      [&]() {
        sgemm_step1_shared_tiled_kernel<16, 16, 8><<<step1_grid, step1_block>>>(
            device_a, device_b, device_c, rows_m, cols_n, depth_k);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTolerance,
      kWarmupIters,
      kRepeatIters));

  results.push_back(benchmark_kernel(
      "step2_thread_tile_1d",
      "第一次显式利用 register blocking",
      [&]() {
        sgemm_step2_thread_tile_1d_kernel<32, 32, 8, 4><<<step2_grid, step2_block>>>(
            device_a, device_b, device_c, rows_m, cols_n, depth_k);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTolerance,
      kWarmupIters,
      kRepeatIters));

  results.push_back(benchmark_kernel(
      "step3_thread_tile_2d",
      "把 shared memory -> register -> outer-product 链条补完整",
      [&]() {
        sgemm_step3_thread_tile_2d_kernel<64, 64, 8, 4, 4><<<step3_grid, step3_block>>>(
            device_a, device_b, device_c, rows_m, cols_n, depth_k);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTolerance,
      kWarmupIters,
      kRepeatIters));

  std::printf("%-24s | %-24s | %12s | %12s | %12s | %8s\n",
              "Kernel",
              "TeachingPoint",
              "MaxAbsErr",
              "Time(ms)",
              "TFLOPS",
              "Status");
  std::printf("-------------------------+--------------------------+--------------+--------------+--------------+----------\n");

  for (const BenchmarkResult &result : results) {
    std::printf("%-24s | %-24s | %12.6g | %12.4f | %12.4f | %8s\n",
                result.name,
                result.teaching_point,
                result.max_abs_error,
                result.avg_time_ms,
                result.tflops,
                result.passed ? "PASS" : "FAIL");
  }

  CHECK_CUDA(cudaFree(device_c));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_a));
  return EXIT_SUCCESS;
}
