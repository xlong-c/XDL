#include "algorithm"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include "float.h"
#include "mma.h"
#include "stdio.h"
#include "stdlib.h"
#include "cublas_v2.h"
#include <cstddef>
#include <cublas_api.h>

void cublas_sgemm(float *A, float *B, float *C, size_t M, size_t N, size_t K) {

  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  static float alpha = 1.0f;
  static float beta = 0.0f;

  cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               M, N, K,
               &alpha,
               A, CUDA_R_32F, M,
               B, CUDA_R_32F, K,
               &beta,
               C, CUDA_R_32F, M,
               CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT);
}

void cublas_sgemm_tf32(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  static float alpha = 1.0f;
  static float beta = 0.0f;
  cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               M, N, K,
               &alpha,
               A, CUDA_R_32F, M,
               B, CUDA_R_32F, K,
               &beta,
               C, CUDA_R_32F, M,
               CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}