#include <__clang_cuda_complex_builtins.h>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<float *>(&(value))[0])

#define SQRT_2_INV 0.7071067811865475f
#define GELU_SCALING_FACTOR 0.7978845608f // sqrt(2/pi)
// gelu: f(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

__device__ __forceinline__ float gelu(float x) {
  return x * 0.5f * (1.0f + erff(x * SQRT_2_INV));
}
__device__ __forceinline__ float gelu_fast(float x) {
  return 0.5f * x *
         (1.0f + tanhf(GELU_SCALING_FACTOR * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float gelu_fast_opt(float x) {
  // 平衡精度和性能
  const float x2 = x * x;
  const float x3 = x2 * x;

  // 使用编译器可能优化的表达式
  const float inner = fmaf(0.044715f, x3, x);
  const float scaled = GELU_SCALING_FACTOR * inner;

  // CUDA的tanhf已经高度优化
  const float tanh_val = tanhf(scaled);

  // 使用对称性减少指令：0.5*x*(1+tanh) = x * (0.5 + 0.5*tanh)
  return x * (0.5f + 0.5f * tanh_val);
}

__global__ void gelu_f32(float *x, float *y, int n) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < n) {
    float4 in = LDST128BITS(x[idx]);
    float4 out;
    out.x = gelu(in.x);
    out.y = gelu(in.y);
    out.z = gelu(in.z);
    out.w = gelu(in.w);
    LDST128BITS(y[idx]) = out;
  }
}