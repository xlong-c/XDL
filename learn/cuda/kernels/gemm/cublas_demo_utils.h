#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error: %s @ %s:%d\n",                       \
                   cudaGetErrorString(err__), __FILE__, __LINE__);             \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status__ = (call);                                          \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "cuBLAS error: %d @ %s:%d\n",                     \
                   static_cast<int>(status__), __FILE__, __LINE__);            \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace cublas_demo {

inline int ceil_div(const int value, const int divisor) {
  return (value + divisor - 1) / divisor;
}

inline size_t ceil_div_size(const size_t value, const size_t divisor) {
  return (value + divisor - 1) / divisor;
}

inline size_t round_up(const size_t value, const size_t multiple) {
  return ceil_div_size(value, multiple) * multiple;
}

inline int parse_positive_int(const char *text, const char *name) {
  char *end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0) {
    std::fprintf(stderr, "非法 %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

inline void ensure_cuda_device() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::fprintf(stderr, "No CUDA device found.\n");
    std::exit(EXIT_FAILURE);
  }
}

inline void print_device_summary() {
  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::printf("device      = %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
}

inline std::vector<float> make_patterned_matrix_col_major(const int rows,
                                                          const int cols,
                                                          const float scale) {
  static constexpr float kPattern[] = {-2.0f, -1.0f, -0.5f, 0.0f,
                                       0.5f,  1.0f,  2.0f,  -1.5f};
  constexpr int kPatternSize = static_cast<int>(sizeof(kPattern) / sizeof(kPattern[0]));

  std::vector<float> values(static_cast<size_t>(rows) * cols);
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      const int index = row + col * rows;
      const int pattern_index = (row * 3 + col * 5 + index) % kPatternSize;
      values[static_cast<size_t>(index)] = kPattern[pattern_index] * scale;
    }
  }
  return values;
}

inline void cpu_gemm_col_major(const std::vector<float> &a,
                               const std::vector<float> &b,
                               std::vector<float> &c,
                               const int m,
                               const int n,
                               const int k,
                               const float alpha = 1.0f,
                               const float beta = 0.0f,
                               const std::vector<float> *c_input = nullptr) {
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        const float a_val = a[static_cast<size_t>(row) + static_cast<size_t>(kk) * m];
        const float b_val = b[static_cast<size_t>(kk) + static_cast<size_t>(col) * k];
        sum += a_val * b_val;
      }
      const size_t index = static_cast<size_t>(row) + static_cast<size_t>(col) * m;
      const float c_prev = c_input == nullptr ? 0.0f : (*c_input)[index];
      c[index] = alpha * sum + beta * c_prev;
    }
  }
}

inline float max_abs_diff(const std::vector<float> &lhs,
                          const std::vector<float> &rhs) {
  float max_err = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    max_err = std::max(max_err, std::fabs(lhs[i] - rhs[i]));
  }
  return max_err;
}

template <typename T> struct PackedType {
  static constexpr size_t packing = 1;
  using type = T;
};

template <> struct PackedType<__nv_fp4_e2m1> {
  static constexpr size_t packing = 2;
  using type = __nv_fp4x2_e2m1;
};

template <typename T>
inline size_t packed_element_count(const size_t num_scalars) {
  return ceil_div_size(num_scalars, PackedType<T>::packing);
}

template <typename T> inline T cast_from_float(const float value);

template <> inline __half cast_from_float<__half>(const float value) {
  return __half(value);
}

template <> inline __nv_bfloat16 cast_from_float<__nv_bfloat16>(const float value) {
  return __nv_bfloat16(value);
}

template <> inline __nv_fp8_e4m3 cast_from_float<__nv_fp8_e4m3>(const float value) {
  return __nv_fp8_e4m3(value);
}

template <> inline __nv_fp8_e5m2 cast_from_float<__nv_fp8_e5m2>(const float value) {
  return __nv_fp8_e5m2(value);
}

template <typename T> inline float cast_to_float(const T value);

template <> inline float cast_to_float<float>(const float value) { return value; }

template <> inline float cast_to_float<__half>(const __half value) {
  return static_cast<float>(value);
}

template <> inline float cast_to_float<__nv_bfloat16>(const __nv_bfloat16 value) {
  return static_cast<float>(value);
}

template <> inline float cast_to_float<__nv_fp8_e4m3>(const __nv_fp8_e4m3 value) {
  return static_cast<float>(value);
}

template <> inline float cast_to_float<__nv_fp8_e5m2>(const __nv_fp8_e5m2 value) {
  return static_cast<float>(value);
}

template <typename T>
inline std::vector<T> quantize_vector(const std::vector<float> &src) {
  std::vector<T> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = cast_from_float<T>(src[i]);
  }
  return dst;
}

template <typename T>
inline std::vector<float> dequantize_vector(const std::vector<T> &src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = cast_to_float<T>(src[i]);
  }
  return dst;
}

inline std::vector<__nv_fp4x2_e2m1>
pack_fp4_vector(const std::vector<float> &src) {
  std::vector<__nv_fp4x2_e2m1> dst(packed_element_count<__nv_fp4_e2m1>(src.size()));
  for (size_t i = 0; i < dst.size(); ++i) {
    const float lo = src[i * 2];
    const float hi = (i * 2 + 1) < src.size() ? src[i * 2 + 1] : 0.0f;
    dst[i] = __nv_fp4x2_e2m1(make_float2(lo, hi));
  }
  return dst;
}

inline std::vector<float>
unpack_fp4_vector(const std::vector<__nv_fp4x2_e2m1> &src,
                  const size_t num_scalars) {
  std::vector<float> dst(num_scalars, 0.0f);
  for (size_t i = 0; i < src.size(); ++i) {
    const float2 pair = static_cast<float2>(src[i]);
    dst[i * 2] = pair.x;
    if (i * 2 + 1 < num_scalars) {
      dst[i * 2 + 1] = pair.y;
    }
  }
  return dst;
}

inline size_t get_scale_tensor_size(const int inner,
                                    const int outer,
                                    const cublasLtMatmulMatrixScale_t scale_mode) {
  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) {
    return 1;
  }

  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ||
      scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
    const size_t scale_vec =
        scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ? 32u : 16u;
    constexpr size_t kBlockCols = 32u;
    constexpr size_t kBlockRows = 4u;
    constexpr size_t kBlockInner = 4u;
    const size_t rows_per_tile = kBlockInner * scale_vec;
    const size_t cols_per_tile = kBlockCols * kBlockRows;
    const size_t scale_rows = round_up(static_cast<size_t>(inner), rows_per_tile) / scale_vec;
    const size_t scale_cols = round_up(static_cast<size_t>(outer), cols_per_tile);
    return scale_rows * scale_cols;
  }

  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F) {
    return static_cast<size_t>(outer);
  }

  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F) {
    return ceil_div_size(static_cast<size_t>(inner), 128u) * static_cast<size_t>(outer);
  }

  if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F) {
    return round_up(ceil_div_size(static_cast<size_t>(inner), 128u), 4u) *
           ceil_div_size(static_cast<size_t>(outer), 128u);
  }

  return 0;
}

template <typename Launch>
inline float benchmark_launch_ms(Launch &&launch,
                                 const int warmup_iters,
                                 const int repeat_iters) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int iter = 0; iter < warmup_iters; ++iter) {
    launch();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  float total_ms = 0.0f;
  for (int iter = 0; iter < repeat_iters; ++iter) {
    CHECK_CUDA(cudaEventRecord(start));
    launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float iter_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
    total_ms += iter_ms;
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return total_ms / static_cast<float>(repeat_iters);
}

inline double compute_tflops(const int m,
                             const int n,
                             const int k,
                             const float time_ms) {
  const double total_flops =
      2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
  return total_flops / (static_cast<double>(time_ms) * 1.0e9);
}

template <typename T>
inline T *malloc_and_copy_to_device(const std::vector<T> &host_data) {
  T *device_ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&device_ptr, host_data.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(device_ptr, host_data.data(), host_data.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return device_ptr;
}

template <typename T> inline void copy_from_device(std::vector<T> &host_data,
                                                   const T *device_ptr) {
  CHECK_CUDA(cudaMemcpy(host_data.data(), device_ptr, host_data.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

} // namespace cublas_demo
