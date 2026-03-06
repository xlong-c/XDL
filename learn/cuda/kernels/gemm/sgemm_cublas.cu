#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error: %s @ %s:%d\n",                       \
                   cudaGetErrorString(err__), __FILE__, __LINE__);             \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status__ = (call);                                          \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "cuBLAS error: %d @ %s:%d\n",                     \
                   static_cast<int>(status__), __FILE__, __LINE__);            \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

struct BenchmarkResult {
  float max_abs_err = 0.0f;
  float avg_time_ms = 0.0f;
  double tflops = 0.0;
  bool passed = false;
};

void fill_matrix(std::vector<float> &data, int mod, int bias) {
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>((static_cast<int>(i % mod) - bias)) * 0.1f;
  }
}

void cpu_sgemm_row_major(const std::vector<float> &a, const std::vector<float> &b,
                         std::vector<float> &c, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        sum += a[row * k + kk] * b[kk * n + col];
      }
      c[row * n + col] = sum;
    }
  }
}

BenchmarkResult benchmark_cublas_sgemm(cublasHandle_t handle, float *d_a,
                                       float *d_b, float *d_c,
                                       std::vector<float> &h_c,
                                       const std::vector<float> &h_ref,
                                       size_t bytes_c, int m, int n, int k,
                                       int warmup_iters, int repeat_iters,
                                       float tolerance) {
  BenchmarkResult result;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  const double total_flops =
      2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // cuBLAS 默认按 column-major 解释矩阵。
  // 对 row-major 的 C = A * B，可利用内存布局等价关系转成：
  //   C^T = B^T * A^T
  // 因而这里以 (N, M, K) 调用，并交换 A/B 的顺序。
  for (int iter = 0; iter < warmup_iters; ++iter) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             d_b, n, d_a, k, &beta, d_c, n));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  for (int iter = 0; iter < repeat_iters; ++iter) {
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             d_b, n, d_a, k, &beta, d_c, n));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float iter_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
    result.avg_time_ms += iter_ms;
  }

  result.avg_time_ms /= static_cast<float>(repeat_iters);
  result.tflops = total_flops / (static_cast<double>(result.avg_time_ms) * 1.0e9);

  CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < h_c.size(); ++i) {
    result.max_abs_err = std::max(result.max_abs_err, std::fabs(h_c[i] - h_ref[i]));
  }
  result.passed = result.max_abs_err <= tolerance;

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return result;
}

int parse_dim(const char *text, const char *name) {
  char *end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0) {
    std::fprintf(stderr, "非法 %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

int main(int argc, char **argv) {
  int m = 512;
  int n = 512;
  int k = 512;
  constexpr float kTolerance = 1e-3f;
  constexpr int kWarmupIters = 5;
  constexpr int kRepeatIters = 20;

  if (argc == 4) {
    m = parse_dim(argv[1], "M");
    n = parse_dim(argv[2], "N");
    k = parse_dim(argv[3], "K");
  } else if (argc != 1) {
    std::fprintf(stderr, "用法: %s [M N K]\n", argv[0]);
    return EXIT_FAILURE;
  }

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::fprintf(stderr, "No CUDA device found.\n");
    return EXIT_FAILURE;
  }

  const size_t size_a = static_cast<size_t>(m) * k;
  const size_t size_b = static_cast<size_t>(k) * n;
  const size_t size_c = static_cast<size_t>(m) * n;
  const size_t bytes_a = size_a * sizeof(float);
  const size_t bytes_b = size_b * sizeof(float);
  const size_t bytes_c = size_c * sizeof(float);

  std::vector<float> h_a(size_a);
  std::vector<float> h_b(size_b);
  std::vector<float> h_c(size_c, 0.0f);
  std::vector<float> h_ref(size_c, 0.0f);

  fill_matrix(h_a, 13, 6);
  fill_matrix(h_b, 17, 8);
  cpu_sgemm_row_major(h_a, h_b, h_ref, m, n, k);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  cublasHandle_t handle = nullptr;

  CHECK_CUDA(cudaMalloc(&d_a, bytes_a));
  CHECK_CUDA(cudaMalloc(&d_b, bytes_b));
  CHECK_CUDA(cudaMalloc(&d_c, bytes_c));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice));

  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

  BenchmarkResult result = benchmark_cublas_sgemm(
      handle, d_a, d_b, d_c, h_c, h_ref, bytes_c, m, n, k, kWarmupIters,
      kRepeatIters, kTolerance);

  std::printf("cuBLAS SGEMM benchmark (row-major)\n");
  std::printf("M=%d, N=%d, K=%d\n", m, n, k);
  std::printf("cublasSgemm args: op(B)=N, op(A)=N, m=%d, n=%d, k=%d\n", n, m,
              k);
  std::printf("max_abs_err = %.6g\n", result.max_abs_err);
  std::printf("avg_time_ms = %.4f\n", result.avg_time_ms);
  std::printf("tflops      = %.4f\n", result.tflops);
  std::printf("status      = %s\n", result.passed ? "PASS" : "FAIL");

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return result.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
