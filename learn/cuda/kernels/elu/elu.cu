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
#define HALF8(value) (reinterpret_cast<float4 *>(&(value))[0])

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
  
  // 使用int4进行128位加载（8个half值）
  int4 data = *reinterpret_cast<int4*>(&x[idx]);
  
  // 从int4中提取8个half值
  // 每个int32包含2个half值（低16位和高16位）
  half x0 = __ushort_as_half(data.x & 0xFFFF);
  half x1 = __ushort_as_half((data.x >> 16) & 0xFFFF);
  half x2 = __ushort_as_half(data.y & 0xFFFF);
  half x3 = __ushort_as_half((data.y >> 16) & 0xFFFF);
  half x4 = __ushort_as_half(data.z & 0xFFFF);
  half x5 = __ushort_as_half((data.z >> 16) & 0xFFFF);
  half x6 = __ushort_as_half(data.w & 0xFFFF);
  half x7 = __ushort_as_half((data.w >> 16) & 0xFFFF);
  
  // 计算ELU
  half y0 = elu_half(x0);
  half y1 = elu_half(x1);
  half y2 = elu_half(x2);
  half y3 = elu_half(x3);
  half y4 = elu_half(x4);
  half y5 = elu_half(x5);
  half y6 = elu_half(x6);
  half y7 = elu_half(x7);
  
  // 将结果打包回int4
  data.x = (__half_as_ushort(y0) & 0xFFFF) | ((__half_as_ushort(y1) & 0xFFFF) << 16);
  data.y = (__half_as_ushort(y2) & 0xFFFF) | ((__half_as_ushort(y3) & 0xFFFF) << 16);
  data.z = (__half_as_ushort(y4) & 0xFFFF) | ((__half_as_ushort(y5) & 0xFFFF) << 16);
  data.w = (__half_as_ushort(y6) & 0xFFFF) | ((__half_as_ushort(y7) & 0xFFFF) << 16);
  
  // 存储结果
  *reinterpret_cast<int4*>(&y[idx]) = data;
}

__global__ void elu_fp16x8_kernel_pack(half *x, half *y, int mask) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= mask) {
    return;
  }

  // 临时寄存器数组，每个线程本地存储8个half值
  half pack_x[8], pack_y[8]; // 8x16 bits = 128 bits

  // 使用HALF8宏一次性加载128位数据
  HALF8(pack_x[0]) = HALF8(x[idx]); // 加载8个half值

  // 循环展开版本：使用#pragma unroll自动展开
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    pack_y[i] = elu_half(pack_x[i]);
  }

  // 使用HALF8宏一次性存储128位数据
  HALF8(y[idx]) = HALF8(pack_y[0]);
}