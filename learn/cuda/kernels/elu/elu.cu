#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
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

__global__ void elu_f32x4_kernel(float *x, float *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mask) {
    y [idx]= elu(x[idx]);
  }
}