#include <cmath>
#include <iostream>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

// Warp size constant
#define WARP_SIZE 32

// Vectorized load/store helpers
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// Zero constants for different precision
#define FP_ZERO 0.0f
#define CUDART_ZERO_FP16 __float2half(0.0f)
#define CUDART_ZERO_BF16 __float2bfloat16(0.0f)

// =============================================================================
// CUDA Kernels
// =============================================================================

// FP32 - Scalar
__global__ void relu_f32_kernel(float *x, float *y, int mask) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < mask) {
    y[idx] = fmaxf(FP_ZERO, x[idx]);
  }
}

// FP32 - Vector4 (float4)
__global__ void relu_f32x4_kernel(float *x, float *y, int mask) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
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

// FP16 - Scalar
__global__ void relu_f16_kernel(half *x, half *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mask) {
    y[idx] = __hmax(CUDART_ZERO_FP16, x[idx]);
  }
}

// FP16 - Vector2 (half2)
__global__ void relu_f16x2_kernel(half *x, half *y, int mask) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < mask) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    reg_y.x = __hmax(CUDART_ZERO_FP16, reg_x.x);
    reg_y.y = __hmax(CUDART_ZERO_FP16, reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

// FP16 - Vector8 (half2 x 4)
__global__ void relu_f16x8_kernel(half *x, half *y, int mask) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < mask) {
    half2 reg_x[4];
    half2 reg_y[4];
    half2 zero_h2 = half2(CUDART_ZERO_FP16, CUDART_ZERO_FP16);
    LDST128BITS(reg_x) = LDST128BITS(x[idx]);

    reg_y[0] = __hmax2(reg_x[0], zero_h2);
    reg_y[1] = __hmax2(reg_x[1], zero_h2);
    reg_y[2] = __hmax2(reg_x[2], zero_h2);
    reg_y[3] = __hmax2(reg_x[3], zero_h2);

    LDST128BITS(y[idx]) = LDST128BITS(reg_y);
  }
}

// FP16 - Packed Vector8 with unroll
__global__ void relu_f16x8_pack_kernel(half *x, half *y, int mask) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= mask)
    return;

  half2 reg_x[4];
  half2 reg_y[4];
  const half2 fp2_zero = half2(CUDART_ZERO_FP16, CUDART_ZERO_FP16);
  LDST128BITS(reg_x) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 4; i++) {
    reg_y[i] = __hmax2(reg_x[i], fp2_zero);
  }

  LDST128BITS(y[idx]) = LDST128BITS(reg_y);
}

// BF16 - Scalar
__global__ void relu_bf16_kernel(nv_bfloat16 *x, nv_bfloat16 *y, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mask) {
    y[idx] = __hmax(CUDART_ZERO_BF16, x[idx]);
  }
}

// BF16 - Vector2 (nv_bfloat162)
__global__ void relu_bf16x2_kernel(nv_bfloat16 *x, nv_bfloat16 *y, int mask) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= mask)
    return;
  const nv_bfloat162 bf2_zero = nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);
  BFLOAT2(y[idx]) = __hmax2(BFLOAT2(x[idx]), bf2_zero);
}

// =============================================================================
// PyTorch Bindings
// =============================================================================

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
  }

#define TORCH_BINDING_RELU(packed_type, th_type, element_type, n_elements) \
void relu_##packed_type(torch::Tensor x, torch::Tensor y) { \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type)) \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type)) \
  const int ndim = x.dim(); \
  int N = 1; \
  for (int i = 0; i < ndim; ++i) { \
    N *= x.size(i); \
  } \
  dim3 block(256 / (n_elements)); \
  dim3 grid((N + 256 - 1) / 256); \
  relu_##packed_type##_kernel<<<grid, block>>>( \
      reinterpret_cast<element_type *>(x.data_ptr()), \
      reinterpret_cast<element_type *>(y.data_ptr()), N); \
}

// Generate PyTorch binding functions for each variant
TORCH_BINDING_RELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_RELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_RELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_RELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_RELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_RELU(f16x8_pack, torch::kHalf, half, 8)
TORCH_BINDING_RELU(bf16, torch::kBFloat16, nv_bfloat16, 1)
TORCH_BINDING_RELU(bf16x2, torch::kBFloat16, nv_bfloat16, 2)

// Register all functions with PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(relu_f32)
  TORCH_BINDING_COMMON_EXTENSION(relu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8_pack)
  TORCH_BINDING_COMMON_EXTENSION(relu_bf16)
  TORCH_BINDING_COMMON_EXTENSION(relu_bf16x2)
}
