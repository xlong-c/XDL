#include <string>

#include "cublas_demo_utils.h"

namespace {

enum class ExPrecision {
  kFp16,
  kBf16,
};

struct ExConfig {
  const char *name;
  cudaDataType input_type;
  cublasComputeType_t compute_type;
  float tolerance;
};

ExPrecision parse_precision(const char *text) {
  const std::string value(text);
  if (value == "fp16") {
    return ExPrecision::kFp16;
  }
  if (value == "bf16") {
    return ExPrecision::kBf16;
  }
  std::fprintf(stderr, "precision 仅支持 fp16 或 bf16, 当前收到: %s\n", text);
  std::exit(EXIT_FAILURE);
}

template <typename InputType>
int run_cublas_gemm_ex(const ExConfig &config, const int m, const int n,
                       const int k) {
  using namespace cublas_demo;

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int kWarmupIters = 5;
  constexpr int kRepeatIters = 30;

  const std::vector<float> h_a_src = make_patterned_matrix_col_major(m, k, 0.25f);
  const std::vector<float> h_b_src = make_patterned_matrix_col_major(k, n, 0.25f);
  const std::vector<InputType> h_a = quantize_vector<InputType>(h_a_src);
  const std::vector<InputType> h_b = quantize_vector<InputType>(h_b_src);
  const std::vector<float> h_a_ref = dequantize_vector(h_a);
  const std::vector<float> h_b_ref = dequantize_vector(h_b);

  std::vector<float> h_c(static_cast<size_t>(m) * n, 0.0f);
  std::vector<float> h_ref(static_cast<size_t>(m) * n, 0.0f);
  cpu_gemm_col_major(h_a_ref, h_b_ref, h_ref, m, n, k, alpha, beta, nullptr);

  InputType *d_a = malloc_and_copy_to_device(h_a);
  InputType *d_b = malloc_and_copy_to_device(h_b);
  float *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_c, h_c.size() * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_c, 0, h_c.size() * sizeof(float)));

  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  const float avg_ms = benchmark_launch_ms(
      [&]() {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                  &alpha, d_a, config.input_type, m, d_b,
                                  config.input_type, k, &beta, d_c,
                                  CUDA_R_32F, m, config.compute_type,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      },
      kWarmupIters, kRepeatIters);

  copy_from_device(h_c, d_c);
  const float max_err = max_abs_diff(h_c, h_ref);

  std::printf("cuBLAS GEMMEx benchmark\n");
  std::printf("precision   = %s\n", config.name);
  std::printf("M=%d, N=%d, K=%d\n", m, n, k);
  print_device_summary();
  std::printf("output_type = fp32\n");
  std::printf("max_abs_err = %.6g\n", max_err);
  std::printf("avg_time_ms = %.4f\n", avg_ms);
  std::printf("tflops      = %.4f\n", compute_tflops(m, n, k, avg_ms));
  std::printf("status      = %s\n",
              max_err <= config.tolerance ? "PASS" : "FAIL");

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return max_err <= config.tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv) {
  using namespace cublas_demo;

  ExPrecision precision = ExPrecision::kFp16;
  int m = 512;
  int n = 512;
  int k = 512;

  if (argc == 5) {
    precision = parse_precision(argv[1]);
    m = parse_positive_int(argv[2], "M");
    n = parse_positive_int(argv[3], "N");
    k = parse_positive_int(argv[4], "K");
  } else if (argc != 1) {
    std::fprintf(stderr, "用法: %s [fp16|bf16 M N K]\n", argv[0]);
    return EXIT_FAILURE;
  }

  ensure_cuda_device();

  if (precision == ExPrecision::kFp16) {
    const ExConfig config{"fp16", CUDA_R_16F, CUBLAS_COMPUTE_32F_FAST_16F,
                          5e-3f};
    return run_cublas_gemm_ex<__half>(config, m, n, k);
  }

  const ExConfig config{"bf16", CUDA_R_16BF, CUBLAS_COMPUTE_32F_FAST_16BF,
                        2e-2f};
  return run_cublas_gemm_ex<__nv_bfloat16>(config, m, n, k);
}
