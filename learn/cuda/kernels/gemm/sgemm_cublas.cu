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

void cublas_sgemm(float *A, float *B, float *C, size_t M, size_t N, size_t K) {

  cublasHandle_t handle = nullptr;
}
