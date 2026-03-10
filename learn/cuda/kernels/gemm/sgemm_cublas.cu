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

/**
 * 标准单精度矩阵乘法 (SGEMM) 实现
 * 计算公式: C = alpha * A * B + beta * C
 */
void cublas_sgemm(float *A, float *B, float *C, size_t M, size_t N, size_t K) {

  cublasHandle_t handle = nullptr;           // 定义 cuBLAS 句柄
  cublasCreate(&handle);                     // 初始化 cuBLAS 上下文
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH); // 设置数学模式为默认（通常指 FP32 精度）

  static float alpha = 1.0f;                 // 标量 alpha = 1.0
  static float beta = 0.0f;                  // 标量 beta = 0.0

  // 使用 cublasGemmEx 进行通用矩阵乘法，支持更多精度配置
  cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,     // A 和 B 矩阵均不转置
               M, N, K,                      // 矩阵维度: M (A行/C行), N (B列/C列), K (A列/B行)
               &alpha,                       // 乘法系数 alpha
               A, CUDA_R_32F, M,             // A 矩阵: 指针, 数据类型(FP32), 主维长度(LDA=M)
               B, CUDA_R_32F, K,             // B 矩阵: 指针, 数据类型(FP32), 主维长度(LDB=K)
               &beta,                        // 加法系数 beta
               C, CUDA_R_32F, M,             // C 矩阵: 指针, 数据类型(FP32), 主维长度(LDC=M)
               CUBLAS_COMPUTE_32F,           // 计算精度: FP32
               CUBLAS_GEMM_DEFAULT);         // 算法选择: 使用默认启发式算法
}

/**
 * 使用 TF32 (Tensor Float 32) 加速的单精度矩阵乘法
 * TF32 在 Tensor Core 上运行，提供类似 FP32 的范围和类似 FP16 的精度（中间累加仍为 FP32）
 */
void cublas_sgemm_tf32(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
  cublasHandle_t handle = nullptr;           // 定义 cuBLAS 句柄
  cublasCreate(&handle);                     // 初始化 cuBLAS 上下文
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH); // 关键：开启 TF32 Tensor Core 加速模式
  static float alpha = 1.0f;                 // 标量 alpha
  static float beta = 0.0f;                  // 标量 beta
  
  cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,     // 矩阵不转置
               M, N, K,                      // 维度信息
               &alpha,                       // alpha
               A, CUDA_R_32F, M,             // 输入 A (FP32)
               B, CUDA_R_32F, K,             // 输入 B (FP32)
               &beta,                        // beta
               C, CUDA_R_32F, M,             // 输出 C (FP32)
               CUBLAS_COMPUTE_32F,           // 内部计算使用 FP32 累加
               CUBLAS_GEMM_DEFAULT_TENSOR_OP); // 明确指令：优先使用 Tensor Core 算子
}
