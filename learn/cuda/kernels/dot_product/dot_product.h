#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// F32 kernels
__global__ void dot_prod_f32_f32_kernel(float *a, float *b, float *y, int N);

__global__ void dot_prod_f32_f32_kernel_v2(float *a, float *b, float *y, int N);

__global__ void dot_prod_f32x4_f32_kernel(float *a, float *b, float *y, int N);

// F16 kernel
__global__ void dot_prod_f16_f32_kernel(half *a, half *b, half *y, int mask);

// BF16 kernel
__global__ void dot_prod_bf16x8_bf16_kernel(__nv_bfloat16 *a, __nv_bfloat16 *b,
                                            __nv_bfloat16 *y, int N);

#endif // DOT_PRODUCT_H
