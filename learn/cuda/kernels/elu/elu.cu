#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define ALPHA 1.0f
// elu :
//  x [x>=0];
// alpha(e^x -1) [x<0];
__device__ __forceinline__ float elu(float x) {
  return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

__device__ __forceinline__ half elu_half(half x) {
  return __hgt(x, __float2half(0.f))
             ? x
             : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.f)));
}

// 标量版本ELU内核
__global__ void elu_f32_scalar_kernel(float *x, float *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mask) {
    y[idx] = elu(x[idx]);
  }
}

// 向量化版本ELU内核（float4）
__global__ void elu_f32x4_kernel(float *x, float *y, int mask) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= mask) {
    return;
  }
  float4 x_reg = FLOAT4(x[idx]);
  float4 y_reg;
  y_reg.x = elu(x_reg.x);
  y_reg.y = elu(x_reg.y);
  y_reg.z = elu(x_reg.z);
  y_reg.w = elu(x_reg.w);
  FLOAT4(y[idx]) = y_reg;
}

// 半精度向量化版本ELU内核（half2）
__global__ void elu_fp16x2_kernel(half *x, half *y, int mask) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < mask) {
    half2 x_reg = HALF2(x[idx]);
    half2 y_reg;
    y_reg.x = elu_half(x_reg.x);
    y_reg.y = elu_half(x_reg.y);
    HALF2(y[idx]) = y_reg;
  }
}
__global__ void elu_fp16x8_kernel(half *x, half *y, int mask) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= mask) {
    return;
  }
  float4 x_reg1 = FLOAT4(x[idx]);
  float4 x_reg2 = FLOAT4(x[idx + 4]);
  float4 y_reg1;
  float4 y_reg2;
  y_reg1.x = __half2float(elu_half(__float2half(x_reg1.x)));
  y_reg1.y = __half2float(elu_half(__float2half(x_reg1.y)));
  y_reg1.z = __half2float(elu_half(__float2half(x_reg1.z)));
  y_reg1.w = __half2float(elu_half(__float2half(x_reg1.w)));
  y_reg2.x = __half2float(elu_half(__float2half(x_reg2.x)));
  y_reg2.y = __half2float(elu_half(__float2half(x_reg2.y)));
  y_reg2.z = __half2float(elu_half(__float2half(x_reg2.z)));
  y_reg2.w = __half2float(elu_half(__float2half(x_reg2.w)));
  FLOAT4(y[idx]) = y_reg1;
  FLOAT4(y[idx + 4]) = y_reg2;
}