#include "cublas_demo_utils.h"

namespace {

int run_fp4_demo(const int m, const int n, const int k) {
  using namespace cublas_demo;

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr float kTolerance = 0.5f;
  constexpr int kWarmupIters = 5;
  constexpr int kRepeatIters = 30;
  constexpr size_t kWorkspaceBytes = 32ull * 1024ull * 1024ull;
  constexpr cublasLtMatmulMatrixScale_t kScaleMode =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

  const std::vector<float> h_a_storage_src =
      make_patterned_matrix_col_major(k, m, 0.5f);
  const std::vector<float> h_b_src = make_patterned_matrix_col_major(k, n, 0.5f);
  const std::vector<__nv_fp4x2_e2m1> h_a = pack_fp4_vector(h_a_storage_src);
  const std::vector<__nv_fp4x2_e2m1> h_b = pack_fp4_vector(h_b_src);
  const std::vector<float> h_a_storage_ref =
      unpack_fp4_vector(h_a, h_a_storage_src.size());
  std::vector<float> h_a_ref(static_cast<size_t>(m) * k, 0.0f);
  for (int kk = 0; kk < k; ++kk) {
    for (int row = 0; row < m; ++row) {
      h_a_ref[static_cast<size_t>(row) + static_cast<size_t>(kk) * m] =
          h_a_storage_ref[static_cast<size_t>(kk) + static_cast<size_t>(row) * k];
    }
  }
  const std::vector<float> h_b_ref = unpack_fp4_vector(h_b, h_b_src.size());

  std::vector<__nv_bfloat16> h_c(static_cast<size_t>(m) * n,
                                 cast_from_float<__nv_bfloat16>(0.0f));
  std::vector<__nv_bfloat16> h_d(static_cast<size_t>(m) * n,
                                 cast_from_float<__nv_bfloat16>(0.0f));
  std::vector<float> h_d_float(h_d.size(), 0.0f);
  std::vector<float> h_ref(h_d.size(), 0.0f);
  cpu_gemm_col_major(h_a_ref, h_b_ref, h_ref, m, n, k, alpha, beta, nullptr);

  const size_t a_scale_size = get_scale_tensor_size(k, m, kScaleMode);
  const size_t b_scale_size = get_scale_tensor_size(k, n, kScaleMode);
  const std::vector<__nv_fp8_e4m3> h_a_scale(
      a_scale_size, cast_from_float<__nv_fp8_e4m3>(1.0f));
  const std::vector<__nv_fp8_e4m3> h_b_scale(
      b_scale_size, cast_from_float<__nv_fp8_e4m3>(1.0f));

  __nv_fp4x2_e2m1 *d_a = malloc_and_copy_to_device(h_a);
  __nv_fp4x2_e2m1 *d_b = malloc_and_copy_to_device(h_b);
  __nv_bfloat16 *d_c = malloc_and_copy_to_device(h_c);
  __nv_bfloat16 *d_d = nullptr;
  CHECK_CUDA(cudaMalloc(&d_d, h_d.size() * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_d, 0, h_d.size() * sizeof(__nv_bfloat16)));
  __nv_fp8_e4m3 *d_a_scale = malloc_and_copy_to_device(h_a_scale);
  __nv_fp8_e4m3 *d_b_scale = malloc_and_copy_to_device(h_b_scale);

  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, kWorkspaceBytes));

  cublasLtHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;

  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F,
                                        CUDA_R_32F));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_TRANSA,
                                              &transa, sizeof(transa)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_TRANSB,
                                              &transb, sizeof(transb)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                              &kScaleMode, sizeof(kScaleMode)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                              &kScaleMode, sizeof(kScaleMode)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale,
      sizeof(d_a_scale)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale,
      sizeof(d_b_scale)));

  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, k, m, k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, k, n, k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, m, n, m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, m, n, m));

  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes,
      sizeof(kWorkspaceBytes)));
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_desc, b_desc,
                                              c_desc, d_desc, preference, 1,
                                              &heuristic, &returned_results));
  if (returned_results == 0) {
    std::fprintf(stderr, "未找到可用的 fp4 cublasLt heuristic.\n");
    return EXIT_FAILURE;
  }

  const float avg_ms = benchmark_launch_ms(
      [&]() {
        CHECK_CUBLAS(cublasLtMatmul(handle, op_desc, &alpha, d_a, a_desc, d_b,
                                    b_desc, &beta, d_c, c_desc, d_d, d_desc,
                                    &heuristic.algo, workspace, kWorkspaceBytes,
                                    0));
      },
      kWarmupIters, kRepeatIters);

  copy_from_device(h_d, d_d);
  h_d_float = dequantize_vector(h_d);
  const float max_err = max_abs_diff(h_d_float, h_ref);

  std::printf("cuBLASLt FP4 GEMM benchmark\n");
  std::printf("input_type  = fp4(e2m1), output_type = bf16\n");
  std::printf("M=%d, N=%d, K=%d\n", m, n, k);
  print_device_summary();
  std::printf("scale_mode  = vec16_ue4m3\n");
  std::printf("max_abs_err = %.6g\n", max_err);
  std::printf("avg_time_ms = %.4f\n", avg_ms);
  std::printf("tflops      = %.4f\n", compute_tflops(m, n, k, avg_ms));
  std::printf("status      = %s\n",
              max_err <= kTolerance ? "PASS" : "FAIL");

  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(d_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(op_desc));
  CHECK_CUBLAS(cublasLtDestroy(handle));

  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_b_scale));
  CHECK_CUDA(cudaFree(d_a_scale));
  CHECK_CUDA(cudaFree(d_d));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a));
  return max_err <= kTolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv) {
  using namespace cublas_demo;

  int m = 64;
  int n = 128;
  int k = 256;
  if (argc == 4) {
    m = parse_positive_int(argv[1], "M");
    n = parse_positive_int(argv[2], "N");
    k = parse_positive_int(argv[3], "K");
  } else if (argc != 1) {
    std::fprintf(stderr, "用法: %s [M N K]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (m % 16 != 0 || n % 16 != 0 || k % 16 != 0) {
    std::fprintf(stderr,
                 "fp4 demo 要求 M/N/K 为 16 的倍数, 当前收到 M=%d N=%d K=%d\n",
                 m, n, k);
    return EXIT_FAILURE;
  }

  ensure_cuda_device();
  return run_fp4_demo(m, n, k);
}
