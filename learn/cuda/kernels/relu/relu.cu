#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Warp size constant
#define WARP_SIZE 32

// Vectorized load/store helpers
#define INT4(value) (reinterpret_cast<int4 *>(&value)[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&value)[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&value)[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&value)[0])
#define LDST123BITS(value) (reinterpret_cast<float4 *>(&value)[0])

__global__ void relu_f32_kernel(float *x, float *y, int mask) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < mask) {
    y[idx] = fmaxf(0.0f, x[idx]);
  }
}

__global__ void relu_f32x4_kernel(float *x, float *y, int mask) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < mask) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(FP_ZERO, reg_x.x);
    reg_y.y = fmaxf(FP_ZERO, reg_x.y);
    reg_y.z = fmaxf(FP_ZERO, reg_x.z);
    reg_y.w = fmaxf(FP_ZERO, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

__global__ void relu_f16_kernel(half *x, half *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mask) {
    y[idx] = __hmax(CUDART_ONE_FP16, x[idx]);
  }
}

__global__ void relu_f16x2_kernel(half *x, half *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mask) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    reg_y.x = __hmax(CUDART_ONE_FP16, reg_x.x);
    reg_y.y = __hmax(CUDART_ONE_FP16, reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

__global__ void relu_f16x8_kernel(half *x, half *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mask) {
    half2 reg_x[4];
    half2 reg_y[4];
    half2 one_h2 = half2(CUDART_ONE_FP16, CUDART_ONE_FP16);
    LDST123BITS(reg_x) = LDST123BITS(x[idx]);

    reg_y[0] = __hmax2(reg_x[0], one_h2);
    reg_y[1] = __hmax2(reg_x[1], one_h2);
    reg_y[2] = __hmax2(reg_x[2], one_h2);
    reg_y[3] = __hmax2(reg_x[3], one_h2);

    LDST123BITS(y[idx]) = LDST123BITS(reg_y);
  }
}

__global__ void relu_f16x8_pack_kernel(half *x, half *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= mask)
    return;
  half2 reg_x[4];
  half2 reg_y[4];
  const half2 fp2_one = half2(CUDART_ONE_FP16, CUDART_ONE_FP16);
  LDST123BITS(reg_x) = LDST123BITS(x[idx]);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    reg_y[i] = __hmax2(reg_x[i], fp2_one);
  }
  LDST123BITS(y[idx]) = LDST123BITS(reg_y);
}

__global__ void relu_bf16_kernel(nv_bfloat16 *x, nv_bfloat16 *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mask) {
    y[idx] = __hmax(CUDART_ONE_BF16, x[idx]);
  }
}

__global__ void relu_bf16x2_kernel(nv_bfloat16 *x, nv_bfloat16 *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= mask)
    return;
  const nv_bfloat162 bf2_one = nv_bfloat162(CUDART_ONE_BF16, CUDART_ONE_BF16);
  BFLOAT2(y[idx]) = __hmax2(BFLOAT2(x[idx]), bf2_one);
}

