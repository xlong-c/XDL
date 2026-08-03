#include <cuda_runtime.h>

#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/int.hpp>
#include <cute/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace cute;

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    cudaError_t error_code__ = (call);                                    \
    if (error_code__ != cudaSuccess) {                                    \
      std::fprintf(stderr, "CUDA error: %s @ %s:%d\n",                    \
                   cudaGetErrorString(error_code__), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

struct BenchmarkResult {
  const char *name = "";
  const char *teaching_point = "";
  float max_abs_error = 0.0f;
  float avg_time_ms = 0.0f;
  double tflops = 0.0;
  bool passed = false;
  bool skipped = false;
};

int parse_positive_int(const char *text, const char *name) {
  char *end_ptr = nullptr;
  const long value = std::strtol(text, &end_ptr, 10);
  if (end_ptr == text || *end_ptr != '\0' || value <= 0 || value > 1'000'000) {
    std::fprintf(stderr, "非法 %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

void print_usage(const char *program_name) {
  std::printf("用法:\n");
  std::printf("  %s\n", program_name);
  std::printf("  %s M N K\n\n", program_name);
  std::printf("默认尺寸: M=N=K=512\n");
  std::printf("当前文件会按 CuTe 学习路径依次运行多个 GEMM level。\n");
  std::printf("注意: 向量化 TiledCopy level 需要 N 和 K 能被 4 整除, 否则会自动 SKIP。\n");
}

void fill_matrix(std::vector<float> &matrix, const int rows, const int cols) {
  for (int row_index = 0; row_index < rows; ++row_index) {
    for (int col_index = 0; col_index < cols; ++col_index) {
      const int linear_index = row_index * cols + col_index;
      const float base_value = static_cast<float>((linear_index % 29) - 14) * 0.0625f;
      const float row_bias = static_cast<float>((row_index % 7) - 3) * 0.015625f;
      const float col_bias = static_cast<float>((col_index % 5) - 2) * 0.0078125f;
      matrix[linear_index] = base_value + row_bias - col_bias;
    }
  }
}

void cpu_sgemm_reference(const std::vector<float> &host_a,
                         const std::vector<float> &host_b,
                         std::vector<float> &host_c,
                         const int rows_m,
                         const int cols_n,
                         const int depth_k) {
  for (int row_index = 0; row_index < rows_m; ++row_index) {
    for (int col_index = 0; col_index < cols_n; ++col_index) {
      float sum_value = 0.0f;
      for (int k_index = 0; k_index < depth_k; ++k_index) {
        sum_value += host_a[row_index * depth_k + k_index] *
                     host_b[k_index * cols_n + col_index];
      }
      host_c[row_index * cols_n + col_index] = sum_value;
    }
  }
}

double compute_tflops(const int rows_m,
                      const int cols_n,
                      const int depth_k,
                      const float avg_time_ms) {
  const double total_flops = 2.0 * static_cast<double>(rows_m) *
                             static_cast<double>(cols_n) *
                             static_cast<double>(depth_k);
  return total_flops / (static_cast<double>(avg_time_ms) * 1.0e9);
}

BenchmarkResult make_skipped_result(const char *kernel_name,
                                    const char *teaching_point) {
  BenchmarkResult result;
  result.name = kernel_name;
  result.teaching_point = teaching_point;
  result.skipped = true;
  result.passed = true;
  return result;
}

template <typename LaunchFunc>
BenchmarkResult benchmark_kernel(const char *kernel_name,
                                 const char *teaching_point,
                                 LaunchFunc launch_kernel,
                                 float *device_c,
                                 std::vector<float> &host_c,
                                 const std::vector<float> &host_reference,
                                 const size_t bytes_c,
                                 const int rows_m,
                                 const int cols_n,
                                 const int depth_k,
                                 const float tolerance,
                                 const int warmup_iters,
                                 const int repeat_iters) {
  BenchmarkResult result;
  result.name = kernel_name;
  result.teaching_point = teaching_point;

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&stop_event));

  CHECK_CUDA(cudaMemset(device_c, 0, bytes_c));
  for (int iter_index = 0; iter_index < warmup_iters; ++iter_index) {
    launch_kernel();
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start_event));
  for (int iter_index = 0; iter_index < repeat_iters; ++iter_index) {
    launch_kernel();
  }
  CHECK_CUDA(cudaEventRecord(stop_event));
  CHECK_CUDA(cudaEventSynchronize(stop_event));
  CHECK_CUDA(cudaGetLastError());

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
  result.avg_time_ms = elapsed_ms / static_cast<float>(repeat_iters);
  result.tflops = compute_tflops(rows_m, cols_n, depth_k, result.avg_time_ms);

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes_c, cudaMemcpyDeviceToHost));
  for (size_t value_index = 0; value_index < host_c.size(); ++value_index) {
    result.max_abs_error = std::max(result.max_abs_error,
                                    std::fabs(host_c[value_index] - host_reference[value_index]));
  }
  result.passed = result.max_abs_error <= tolerance;

  CHECK_CUDA(cudaEventDestroy(stop_event));
  CHECK_CUDA(cudaEventDestroy(start_event));
  return result;
}

template <typename PredTensor, typename CoordTensor>
CUTE_HOST_DEVICE void fill_predicate_2d(PredTensor &pred,
                                        const CoordTensor &coord,
                                        const int limit_0,
                                        const int limit_1) {
  CUTE_UNROLL
  for (int i = 0; i < size<0>(pred); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(pred); ++j) {
      pred(i, j) = get<0>(coord(i, j)) < limit_0 && get<1>(coord(i, j)) < limit_1;
    }
  }
}

template <typename PredTensor, typename CoordTensor>
CUTE_HOST_DEVICE void fill_predicate_3d(PredTensor &pred,
                                        const CoordTensor &coord,
                                        const int limit_0,
                                        const int limit_1) {
  CUTE_UNROLL
  for (int i = 0; i < size<0>(pred); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(pred); ++j) {
      CUTE_UNROLL
      for (int k = 0; k < size<2>(pred); ++k) {
        pred(i, j, k) = get<0>(coord(i, j, k)) < limit_0 &&
                        get<1>(coord(i, j, k)) < limit_1;
      }
    }
  }
}

template <typename PredTensor, typename CoordTensor>
CUTE_HOST_DEVICE void fill_vector_copy_predicate_2d(PredTensor &pred,
                                                    const CoordTensor &coord,
                                                    const int limit_0,
                                                    const int limit_1) {
  CUTE_UNROLL
  for (int i = 0; i < size<0>(pred); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(pred); ++j) {
      pred(i, j) = get<0>(coord(Int<0>{}, i, j)) < limit_0 &&
                   get<1>(coord(Int<0>{}, i, j)) < limit_1;
    }
  }
}

__global__ void sgemm_cute_view_naive_kernel(const float *matrix_a,
                                             const float *matrix_b,
                                             float *matrix_c,
                                             const int rows_m,
                                             const int cols_n,
                                             const int depth_k) {
  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int global_row = blockIdx.y * blockDim.y + local_row;
  const int global_col = blockIdx.x * blockDim.x + local_col;

  auto tensor_a = make_tensor(
      make_gmem_ptr(matrix_a),
      make_layout(make_shape(rows_m, depth_k), GenRowMajor{}));
  auto tensor_b = make_tensor(
      make_gmem_ptr(matrix_b),
      make_layout(make_shape(depth_k, cols_n), GenRowMajor{}));
  auto tensor_c = make_tensor(
      make_gmem_ptr(matrix_c),
      make_layout(make_shape(rows_m, cols_n), GenRowMajor{}));

  if (global_row >= rows_m || global_col >= cols_n) {
    return;
  }

  float accumulator = 0.0f;
  for (int k_index = 0; k_index < depth_k; ++k_index) {
    accumulator = __fmaf_rn(tensor_a(global_row, k_index),
                            tensor_b(k_index, global_col),
                            accumulator);
  }

  tensor_c(global_row, global_col) = accumulator;
}

template <int BlockRowsM = 16, int BlockColsN = 16, int BlockDepthK = 16>
__global__ void sgemm_cute_shared_tiled_kernel(const float *matrix_a,
                                               const float *matrix_b,
                                               float *matrix_c,
                                               const int rows_m,
                                               const int cols_n,
                                               const int depth_k) {
  static_assert(BlockRowsM == BlockColsN && BlockColsN == BlockDepthK,
                "当前教学版 CuTe GEMM 使用 16x16x16 方形 tile。\n");

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int global_row = blockIdx.y * BlockRowsM + local_row;
  const int global_col = blockIdx.x * BlockColsN + local_col;

  auto matrix_a_tensor = make_tensor(
      make_gmem_ptr(matrix_a),
      make_layout(make_shape(rows_m, depth_k), GenRowMajor{}));
  auto matrix_b_tensor = make_tensor(
      make_gmem_ptr(matrix_b),
      make_layout(make_shape(depth_k, cols_n), GenRowMajor{}));
  auto matrix_c_tensor = make_tensor(
      make_gmem_ptr(matrix_c),
      make_layout(make_shape(rows_m, cols_n), GenRowMajor{}));

  auto output_tile = local_tile(matrix_c_tensor,
                                make_shape(Int<BlockRowsM>{}, Int<BlockColsN>{}),
                                make_coord(blockIdx.y, blockIdx.x));

  __shared__ float shared_a_storage[BlockRowsM * BlockDepthK];
  __shared__ float shared_b_storage[BlockDepthK * BlockColsN];

  auto shared_a_tensor = make_tensor(
      make_smem_ptr(shared_a_storage),
      make_layout(make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                  GenRowMajor{}));
  auto shared_b_tensor = make_tensor(
      make_smem_ptr(shared_b_storage),
      make_layout(make_shape(Int<BlockDepthK>{}, Int<BlockColsN>{}),
                  GenRowMajor{}));

  float accumulator = 0.0f;
  const int k_tiles = (depth_k + BlockDepthK - 1) / BlockDepthK;

  for (int k_tile_index = 0; k_tile_index < k_tiles; ++k_tile_index) {
    auto a_tile = local_tile(matrix_a_tensor,
                             make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                             make_coord(blockIdx.y, k_tile_index));
    auto b_tile = local_tile(matrix_b_tensor,
                             make_shape(Int<BlockDepthK>{}, Int<BlockColsN>{}),
                             make_coord(k_tile_index, blockIdx.x));

    const int global_k_for_a = k_tile_index * BlockDepthK + local_col;
    const int global_k_for_b = k_tile_index * BlockDepthK + local_row;

    shared_a_tensor(local_row, local_col) =
        (global_row < rows_m && global_k_for_a < depth_k)
            ? a_tile(local_row, local_col)
            : 0.0f;
    shared_b_tensor(local_row, local_col) =
        (global_k_for_b < depth_k && global_col < cols_n)
            ? b_tile(local_row, local_col)
            : 0.0f;

    __syncthreads();

#pragma unroll
    for (int k_index = 0; k_index < BlockDepthK; ++k_index) {
      accumulator = __fmaf_rn(shared_a_tensor(local_row, k_index),
                              shared_b_tensor(k_index, local_col),
                              accumulator);
    }

    __syncthreads();
  }

  if (global_row < rows_m && global_col < cols_n) {
    output_tile(local_row, local_col) = accumulator;
  }
}

template <int BlockRowsM = 16,
          int BlockColsN = 16,
          int BlockDepthK = 16,
          typename ThreadLayoutA,
          typename ThreadLayoutB>
__global__ void sgemm_cute_partition_copy_kernel(const float *matrix_a,
                                                 const float *matrix_b,
                                                 float *matrix_c,
                                                 const int rows_m,
                                                 const int cols_n,
                                                 const int depth_k,
                                                 ThreadLayoutA thread_layout_a,
                                                 ThreadLayoutB thread_layout_b) {
  const int thread_index = threadIdx.x;
  const int local_row = thread_index / BlockColsN;
  const int local_col = thread_index % BlockColsN;
  const int global_row = blockIdx.y * BlockRowsM + local_row;
  const int global_col = blockIdx.x * BlockColsN + local_col;

  auto matrix_a_tensor = make_tensor(
      make_gmem_ptr(matrix_a),
      make_layout(make_shape(rows_m, depth_k), GenRowMajor{}));
  auto matrix_b_tensor = make_tensor(
      make_gmem_ptr(matrix_b),
      make_layout(make_shape(depth_k, cols_n), GenRowMajor{}));
  auto matrix_c_tensor = make_tensor(
      make_gmem_ptr(matrix_c),
      make_layout(make_shape(rows_m, cols_n), GenRowMajor{}));

  auto output_tile = local_tile(matrix_c_tensor,
                                make_shape(Int<BlockRowsM>{}, Int<BlockColsN>{}),
                                make_coord(blockIdx.y, blockIdx.x));
  auto coord_a_tensor = make_identity_tensor(matrix_a_tensor.shape());
  auto coord_b_tensor = make_identity_tensor(matrix_b_tensor.shape());

  __shared__ float shared_a_storage[BlockRowsM * BlockDepthK];
  __shared__ float shared_b_storage[BlockDepthK * BlockColsN];

  auto shared_a_tensor = make_tensor(
      make_smem_ptr(shared_a_storage),
      make_layout(make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                  GenRowMajor{}));
  auto shared_b_tensor = make_tensor(
      make_smem_ptr(shared_b_storage),
      make_layout(make_shape(Int<BlockDepthK>{}, Int<BlockColsN>{}),
                  GenRowMajor{}));

  float accumulator = 0.0f;
  const int k_tiles = (depth_k + BlockDepthK - 1) / BlockDepthK;

  for (int k_tile_index = 0; k_tile_index < k_tiles; ++k_tile_index) {
    auto a_tile = local_tile(matrix_a_tensor,
                             make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                             make_coord(blockIdx.y, k_tile_index));
    auto b_tile = local_tile(matrix_b_tensor,
                             make_shape(Int<BlockDepthK>{}, Int<BlockColsN>{}),
                             make_coord(k_tile_index, blockIdx.x));
    auto coord_a_tile = local_tile(coord_a_tensor,
                                   make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                                   make_coord(blockIdx.y, k_tile_index));
    auto coord_b_tile = local_tile(coord_b_tensor,
                                   make_shape(Int<BlockDepthK>{}, Int<BlockColsN>{}),
                                   make_coord(k_tile_index, blockIdx.x));

    Tensor thread_a_global = local_partition(a_tile, thread_layout_a, thread_index);
    Tensor thread_b_global = local_partition(b_tile, thread_layout_b, thread_index);
    Tensor thread_a_shared = local_partition(shared_a_tensor, thread_layout_a, thread_index);
    Tensor thread_b_shared = local_partition(shared_b_tensor, thread_layout_b, thread_index);
    Tensor thread_a_coord = local_partition(coord_a_tile, thread_layout_a, thread_index);
    Tensor thread_b_coord = local_partition(coord_b_tile, thread_layout_b, thread_index);

    clear(thread_a_shared);
    clear(thread_b_shared);

    Tensor thread_a_pred = make_tensor<bool>(thread_a_coord.shape());
    Tensor thread_b_pred = make_tensor<bool>(thread_b_coord.shape());
    fill_predicate_2d(thread_a_pred, thread_a_coord, rows_m, depth_k);
    fill_predicate_2d(thread_b_pred, thread_b_coord, depth_k, cols_n);

    copy_if(thread_a_pred, thread_a_global, thread_a_shared);
    copy_if(thread_b_pred, thread_b_global, thread_b_shared);

    __syncthreads();

#pragma unroll
    for (int k_index = 0; k_index < BlockDepthK; ++k_index) {
      accumulator = __fmaf_rn(shared_a_tensor(local_row, k_index),
                              shared_b_tensor(k_index, local_col),
                              accumulator);
    }

    __syncthreads();
  }

  if (global_row < rows_m && global_col < cols_n) {
    output_tile(local_row, local_col) = accumulator;
  }
}

template <int BlockRowsM = 128,
          int BlockColsN = 128,
          int BlockDepthK = 8,
          typename ThreadLayoutA,
          typename ThreadLayoutB,
          typename ThreadLayoutC,
          typename SmemLayoutA,
          typename SmemLayoutB>
__global__ void sgemm_cute_threadlayout_kernel(const float *matrix_a,
                                               const float *matrix_b,
                                               float *matrix_c,
                                               const int rows_m,
                                               const int cols_n,
                                               const int depth_k,
                                               ThreadLayoutA thread_layout_a,
                                               ThreadLayoutB thread_layout_b,
                                               ThreadLayoutC thread_layout_c,
                                               SmemLayoutA smem_layout_a,
                                               SmemLayoutB smem_layout_b) {
  const int thread_index = threadIdx.x;

  auto tensor_a = make_tensor(
      make_gmem_ptr(matrix_a),
      make_layout(make_shape(rows_m, depth_k), make_stride(depth_k, Int<1>{})));
  auto tensor_b = make_tensor(
      make_gmem_ptr(matrix_b),
      make_layout(make_shape(cols_n, depth_k), make_stride(Int<1>{}, cols_n)));
  auto tensor_c = make_tensor(
      make_gmem_ptr(matrix_c),
      make_layout(make_shape(rows_m, cols_n), make_stride(cols_n, Int<1>{})));

  auto output_tile = local_tile(tensor_c,
                                make_shape(Int<BlockRowsM>{}, Int<BlockColsN>{}),
                                make_coord(blockIdx.y, blockIdx.x));
  auto coord_a_tensor = make_identity_tensor(tensor_a.shape());
  auto coord_b_tensor = make_identity_tensor(tensor_b.shape());
  auto coord_c_tensor = local_tile(make_identity_tensor(tensor_c.shape()),
                                   make_shape(Int<BlockRowsM>{}, Int<BlockColsN>{}),
                                   make_coord(blockIdx.y, blockIdx.x));

  constexpr int kSmemAStorage = BlockRowsM * (BlockDepthK + 1);
  constexpr int kSmemBStorage = BlockColsN * (BlockDepthK + 1);
  __shared__ float shared_a_storage[kSmemAStorage];
  __shared__ float shared_b_storage[kSmemBStorage];

  auto shared_a_tensor = make_tensor(make_smem_ptr(shared_a_storage), smem_layout_a);
  auto shared_b_tensor = make_tensor(make_smem_ptr(shared_b_storage), smem_layout_b);

  Tensor thread_compute_a =
      local_partition(shared_a_tensor, thread_layout_c, thread_index, Step<_1, X>{});
  Tensor thread_compute_b =
      local_partition(shared_b_tensor, thread_layout_c, thread_index, Step<X, _1>{});
  Tensor thread_output =
      local_partition(output_tile, thread_layout_c, thread_index, Step<_1, _1>{});
  Tensor thread_output_coord =
      local_partition(coord_c_tensor, thread_layout_c, thread_index, Step<_1, _1>{});
  Tensor thread_accumulator = make_tensor_like(thread_output);
  clear(thread_accumulator);

  const int k_tiles = (depth_k + BlockDepthK - 1) / BlockDepthK;
  for (int k_tile_index = 0; k_tile_index < k_tiles; ++k_tile_index) {
    auto a_tile = local_tile(tensor_a,
                             make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                             make_coord(blockIdx.y, k_tile_index));
    auto b_tile = local_tile(tensor_b,
                             make_shape(Int<BlockColsN>{}, Int<BlockDepthK>{}),
                             make_coord(blockIdx.x, k_tile_index));
    auto coord_a_tile = local_tile(coord_a_tensor,
                                   make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                                   make_coord(blockIdx.y, k_tile_index));
    auto coord_b_tile = local_tile(coord_b_tensor,
                                   make_shape(Int<BlockColsN>{}, Int<BlockDepthK>{}),
                                   make_coord(blockIdx.x, k_tile_index));

    Tensor thread_a_global = local_partition(a_tile, thread_layout_a, thread_index);
    Tensor thread_a_shared = local_partition(shared_a_tensor, thread_layout_a, thread_index);
    Tensor thread_a_coord = local_partition(coord_a_tile, thread_layout_a, thread_index);
    Tensor thread_b_global = local_partition(b_tile, thread_layout_b, thread_index);
    Tensor thread_b_shared = local_partition(shared_b_tensor, thread_layout_b, thread_index);
    Tensor thread_b_coord = local_partition(coord_b_tile, thread_layout_b, thread_index);

    clear(thread_a_shared);
    clear(thread_b_shared);

    Tensor thread_a_pred = make_tensor<bool>(thread_a_coord.shape());
    Tensor thread_b_pred = make_tensor<bool>(thread_b_coord.shape());
    fill_predicate_2d(thread_a_pred, thread_a_coord, rows_m, depth_k);
    fill_predicate_2d(thread_b_pred, thread_b_coord, cols_n, depth_k);

    copy_if(thread_a_pred, thread_a_global, thread_a_shared);
    copy_if(thread_b_pred, thread_b_global, thread_b_shared);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    gemm(thread_compute_a, thread_compute_b, thread_accumulator);
    __syncthreads();
  }

  Tensor thread_output_pred =
      make_tensor<bool>(thread_output_coord.shape());
  fill_predicate_2d(thread_output_pred, thread_output_coord, rows_m, cols_n);
  copy_if(thread_output_pred, thread_accumulator, thread_output);
}

template <int BlockRowsM = 128,
          int BlockColsN = 128,
          int BlockDepthK = 8,
          typename GmemTiledCopyA,
          typename GmemTiledCopyB,
          typename ThreadLayoutC,
          typename SmemLayoutA,
          typename SmemLayoutB>
__global__ void sgemm_cute_tiledcopy_kernel(const float *matrix_a,
                                            const float *matrix_b,
                                            float *matrix_c,
                                            const int rows_m,
                                            const int cols_n,
                                            const int depth_k,
                                            GmemTiledCopyA copy_a,
                                            GmemTiledCopyB copy_b,
                                            ThreadLayoutC thread_layout_c,
                                            SmemLayoutA smem_layout_a,
                                            SmemLayoutB smem_layout_b) {
  const int thread_index = threadIdx.x;

  auto tensor_a = make_tensor(
      make_gmem_ptr(matrix_a),
      make_layout(make_shape(rows_m, depth_k), make_stride(depth_k, Int<1>{})));
  auto tensor_b = make_tensor(
      make_gmem_ptr(matrix_b),
      make_layout(make_shape(cols_n, depth_k), make_stride(Int<1>{}, cols_n)));
  auto tensor_c = make_tensor(
      make_gmem_ptr(matrix_c),
      make_layout(make_shape(rows_m, cols_n), make_stride(cols_n, Int<1>{})));

  auto output_tile = local_tile(tensor_c,
                                make_shape(Int<BlockRowsM>{}, Int<BlockColsN>{}),
                                make_coord(blockIdx.y, blockIdx.x));
  auto coord_a_tensor = make_identity_tensor(tensor_a.shape());
  auto coord_b_tensor = make_identity_tensor(tensor_b.shape());
  auto coord_c_tensor = local_tile(make_identity_tensor(tensor_c.shape()),
                                   make_shape(Int<BlockRowsM>{}, Int<BlockColsN>{}),
                                   make_coord(blockIdx.y, blockIdx.x));

  constexpr int kSmemAStorage = BlockRowsM * (BlockDepthK + 1);
  constexpr int kSmemBStorage = BlockColsN * (BlockDepthK + 1);
  __shared__ float shared_a_storage[kSmemAStorage];
  __shared__ float shared_b_storage[kSmemBStorage];

  auto shared_a_tensor = make_tensor(make_smem_ptr(shared_a_storage), smem_layout_a);
  auto shared_b_tensor = make_tensor(make_smem_ptr(shared_b_storage), smem_layout_b);

  auto thread_copy_a = copy_a.get_slice(thread_index);
  auto thread_copy_b = copy_b.get_slice(thread_index);

  Tensor thread_compute_a =
      local_partition(shared_a_tensor, thread_layout_c, thread_index, Step<_1, X>{});
  Tensor thread_compute_b =
      local_partition(shared_b_tensor, thread_layout_c, thread_index, Step<X, _1>{});
  Tensor thread_output =
      local_partition(output_tile, thread_layout_c, thread_index, Step<_1, _1>{});
  Tensor thread_output_coord =
      local_partition(coord_c_tensor, thread_layout_c, thread_index, Step<_1, _1>{});
  Tensor thread_accumulator = make_tensor_like(thread_output);
  clear(thread_accumulator);

  const int k_tiles = (depth_k + BlockDepthK - 1) / BlockDepthK;
  for (int k_tile_index = 0; k_tile_index < k_tiles; ++k_tile_index) {
    auto a_tile = local_tile(tensor_a,
                             make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                             make_coord(blockIdx.y, k_tile_index));
    auto b_tile = local_tile(tensor_b,
                             make_shape(Int<BlockColsN>{}, Int<BlockDepthK>{}),
                             make_coord(blockIdx.x, k_tile_index));
    auto coord_a_tile = local_tile(coord_a_tensor,
                                   make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                                   make_coord(blockIdx.y, k_tile_index));
    auto coord_b_tile = local_tile(coord_b_tensor,
                                   make_shape(Int<BlockColsN>{}, Int<BlockDepthK>{}),
                                   make_coord(blockIdx.x, k_tile_index));

    Tensor thread_a_global = thread_copy_a.partition_S(a_tile);
    Tensor thread_a_shared = thread_copy_a.partition_D(shared_a_tensor);
    Tensor thread_a_coord = thread_copy_a.partition_S(coord_a_tile);
    Tensor thread_b_global = thread_copy_b.partition_S(b_tile);
    Tensor thread_b_shared = thread_copy_b.partition_D(shared_b_tensor);
    Tensor thread_b_coord = thread_copy_b.partition_S(coord_b_tile);

    clear(thread_a_shared);
    clear(thread_b_shared);

    Tensor thread_a_pred = make_tensor<bool>(thread_a_coord.shape());
    Tensor thread_b_pred = make_tensor<bool>(thread_b_coord.shape());
    fill_predicate_3d(thread_a_pred, thread_a_coord, rows_m, depth_k);
    fill_predicate_3d(thread_b_pred, thread_b_coord, cols_n, depth_k);

    copy_if(thread_a_pred, thread_a_global, thread_a_shared);
    copy_if(thread_b_pred, thread_b_global, thread_b_shared);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    gemm(thread_compute_a, thread_compute_b, thread_accumulator);
    __syncthreads();
  }

  Tensor thread_output_pred =
      make_tensor<bool>(thread_output_coord.shape());
  fill_predicate_2d(thread_output_pred, thread_output_coord, rows_m, cols_n);
  copy_if(thread_output_pred, thread_accumulator, thread_output);
}

template <int BlockRowsM = 128,
          int BlockColsN = 128,
          int BlockDepthK = 8,
          typename GmemTiledCopyA,
          typename GmemTiledCopyB,
          typename SmemLayoutA,
          typename SmemLayoutB>
__global__ void sgemm_cute_tiledmma_kernel(const float *matrix_a,
                                           const float *matrix_b,
                                           float *matrix_c,
                                           const int rows_m,
                                           const int cols_n,
                                           const int depth_k,
                                           GmemTiledCopyA copy_a,
                                           GmemTiledCopyB copy_b,
                                           SmemLayoutA smem_layout_a,
                                           SmemLayoutB smem_layout_b) {
  const int thread_index = threadIdx.x;

  auto tensor_a = make_tensor(
      make_gmem_ptr(matrix_a),
      make_layout(make_shape(rows_m, depth_k), make_stride(depth_k, Int<1>{})));
  auto tensor_b = make_tensor(
      make_gmem_ptr(matrix_b),
      make_layout(make_shape(cols_n, depth_k), make_stride(Int<1>{}, cols_n)));
  auto tensor_c = make_tensor(
      make_gmem_ptr(matrix_c),
      make_layout(make_shape(rows_m, cols_n), make_stride(cols_n, Int<1>{})));

  auto output_tile = local_tile(tensor_c,
                                make_shape(Int<BlockRowsM>{}, Int<BlockColsN>{}),
                                make_coord(blockIdx.y, blockIdx.x));
  auto coord_a_tensor = make_identity_tensor(tensor_a.shape());
  auto coord_b_tensor = make_identity_tensor(tensor_b.shape());
  auto coord_c_tensor = local_tile(make_identity_tensor(tensor_c.shape()),
                                   make_shape(Int<BlockRowsM>{}, Int<BlockColsN>{}),
                                   make_coord(blockIdx.y, blockIdx.x));

  constexpr int kSmemAStorage = BlockRowsM * (BlockDepthK + 1);
  constexpr int kSmemBStorage = BlockColsN * (BlockDepthK + 1);
  __shared__ float shared_a_storage[kSmemAStorage];
  __shared__ float shared_b_storage[kSmemBStorage];

  auto shared_a_tensor = make_tensor(make_smem_ptr(shared_a_storage), smem_layout_a);
  auto shared_b_tensor = make_tensor(make_smem_ptr(shared_b_storage), smem_layout_b);

  auto thread_copy_a = copy_a.get_slice(thread_index);
  auto thread_copy_b = copy_b.get_slice(thread_index);
  auto tiled_mma = make_tiled_mma(UniversalFMA<float, float, float>{},
                                  Layout<Shape<_16, _16, _1>>{});
  auto thread_mma = tiled_mma.get_slice(thread_index);

  Tensor thread_output = thread_mma.partition_C(output_tile);
  Tensor thread_output_coord = thread_mma.partition_C(coord_c_tensor);
  Tensor thread_accumulator = thread_mma.make_fragment_C(thread_output);
  clear(thread_accumulator);

  const int k_tiles = (depth_k + BlockDepthK - 1) / BlockDepthK;
  for (int k_tile_index = 0; k_tile_index < k_tiles; ++k_tile_index) {
    auto a_tile = local_tile(tensor_a,
                             make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                             make_coord(blockIdx.y, k_tile_index));
    auto b_tile = local_tile(tensor_b,
                             make_shape(Int<BlockColsN>{}, Int<BlockDepthK>{}),
                             make_coord(blockIdx.x, k_tile_index));
    auto coord_a_tile = local_tile(coord_a_tensor,
                                   make_shape(Int<BlockRowsM>{}, Int<BlockDepthK>{}),
                                   make_coord(blockIdx.y, k_tile_index));
    auto coord_b_tile = local_tile(coord_b_tensor,
                                   make_shape(Int<BlockColsN>{}, Int<BlockDepthK>{}),
                                   make_coord(blockIdx.x, k_tile_index));

    Tensor thread_a_global = thread_copy_a.partition_S(a_tile);
    Tensor thread_a_shared = thread_copy_a.partition_D(shared_a_tensor);
    Tensor thread_a_coord = thread_copy_a.partition_S(coord_a_tile);
    Tensor thread_b_global = thread_copy_b.partition_S(b_tile);
    Tensor thread_b_shared = thread_copy_b.partition_D(shared_b_tensor);
    Tensor thread_b_coord = thread_copy_b.partition_S(coord_b_tile);

    clear(thread_a_shared);
    clear(thread_b_shared);

    Tensor thread_a_pred = make_tensor<bool>(thread_a_coord.shape());
    Tensor thread_b_pred = make_tensor<bool>(thread_b_coord.shape());
    fill_predicate_3d(thread_a_pred, thread_a_coord, rows_m, depth_k);
    fill_predicate_3d(thread_b_pred, thread_b_coord, cols_n, depth_k);

    copy_if(thread_a_pred, thread_a_global, thread_a_shared);
    copy_if(thread_b_pred, thread_b_global, thread_b_shared);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    cooperative_gemm(thread_index, tiled_mma, shared_a_tensor, shared_b_tensor, thread_accumulator);
    __syncthreads();
  }

  Tensor thread_output_pred =
      make_tensor<bool>(thread_output_coord.shape());
  fill_predicate_3d(thread_output_pred, thread_output_coord, rows_m, cols_n);
  copy_if(thread_output_pred, thread_accumulator, thread_output);
}

bool support_vectorized_tiledcopy(const int cols_n, const int depth_k) {
  (void)cols_n;
  (void)depth_k;
  return true;
}

int main(int argc, char **argv) {
  constexpr int kDefaultSize = 512;
  constexpr int kWarmupIters = 3;
  constexpr int kRepeatIters = 10;
  constexpr float kTolerance = 1e-3f;

  constexpr int kSmallBM = 16;
  constexpr int kSmallBN = 16;
  constexpr int kSmallBK = 16;
  constexpr int kBigBM = 128;
  constexpr int kBigBN = 128;
  constexpr int kBigBK = 8;

  int rows_m = kDefaultSize;
  int cols_n = kDefaultSize;
  int depth_k = kDefaultSize;

  if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                    std::strcmp(argv[1], "-h") == 0)) {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  if (argc == 4) {
    rows_m = parse_positive_int(argv[1], "M");
    cols_n = parse_positive_int(argv[2], "N");
    depth_k = parse_positive_int(argv[3], "K");
  } else if (argc != 1) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::fprintf(stderr, "未检测到 CUDA 设备。\n");
    return EXIT_FAILURE;
  }

  const size_t size_a = static_cast<size_t>(rows_m) * depth_k;
  const size_t size_b = static_cast<size_t>(depth_k) * cols_n;
  const size_t size_c = static_cast<size_t>(rows_m) * cols_n;
  const size_t bytes_a = size_a * sizeof(float);
  const size_t bytes_b = size_b * sizeof(float);
  const size_t bytes_c = size_c * sizeof(float);

  std::vector<float> host_a(size_a, 0.0f);
  std::vector<float> host_b(size_b, 0.0f);
  std::vector<float> host_c(size_c, 0.0f);
  std::vector<float> host_reference(size_c, 0.0f);

  fill_matrix(host_a, rows_m, depth_k);
  fill_matrix(host_b, depth_k, cols_n);
  cpu_sgemm_reference(host_a, host_b, host_reference, rows_m, cols_n, depth_k);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a, bytes_a));
  CHECK_CUDA(cudaMalloc(&device_b, bytes_b));
  CHECK_CUDA(cudaMalloc(&device_c, bytes_c));
  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes_b, cudaMemcpyHostToDevice));

  const dim3 naive_block(16, 16);
  const dim3 naive_grid((cols_n + naive_block.x - 1) / naive_block.x,
                        (rows_m + naive_block.y - 1) / naive_block.y);
  const dim3 small_block_2d(kSmallBN, kSmallBM);
  const dim3 small_grid((cols_n + kSmallBN - 1) / kSmallBN,
                        (rows_m + kSmallBM - 1) / kSmallBM);
  auto small_partition_a = make_layout(make_shape(Int<kSmallBM>{}, Int<kSmallBK>{}),
                                       GenRowMajor{});
  auto small_partition_b = make_layout(make_shape(Int<kSmallBK>{}, Int<kSmallBN>{}),
                                       GenRowMajor{});
  const dim3 small_partition_block(size(small_partition_a));

  const dim3 big_grid((cols_n + kBigBN - 1) / kBigBN,
                      (rows_m + kBigBM - 1) / kBigBM);
  auto thread_layout_a = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto thread_layout_b = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto thread_layout_c = make_layout(make_shape(Int<16>{}, Int<16>{}));
  auto smem_layout_a = make_layout(make_shape(Int<kBigBM>{}, Int<kBigBK>{}),
                                   make_stride(Int<kBigBK>{}, Int<1>{}));
  auto smem_layout_b = make_layout(make_shape(Int<kBigBN>{}, Int<kBigBK>{}),
                                   make_stride(Int<1>{}, Int<kBigBN>{}));
  auto copy_a = make_tiled_copy(Copy_Atom<DefaultCopy, float>{},
                                thread_layout_a,
                                make_layout(make_shape(Int<1>{}, Int<1>{})));
  auto copy_b = make_tiled_copy(Copy_Atom<DefaultCopy, float>{},
                                thread_layout_b,
                                make_layout(make_shape(Int<1>{}, Int<1>{})));
  static_assert(size(thread_layout_a) == size(thread_layout_b));
  static_assert(size(thread_layout_a) == size(thread_layout_c));
  static_assert(size(copy_a) == size(copy_b));
  const dim3 big_block(size(thread_layout_c));
  const bool can_run_vectorized_levels = support_vectorized_tiledcopy(cols_n, depth_k);

  std::printf("CuTe 版 SGEMM 分层 demo\n");
  std::printf("矩阵尺寸: M=%d, N=%d, K=%d\n", rows_m, cols_n, depth_k);
  std::printf("分层路线:\n");
  std::printf("  L0 view_only_naive -> L1 shared_tiled -> L2 partition_copy\n");
  std::printf("  L3 threadlayout_gemm -> L4 tiledcopy -> L5 tiledmma_universalfma\n");
  std::printf("\n");

  std::vector<BenchmarkResult> results;
  results.push_back(benchmark_kernel(
      "L0_view_naive_f32",
      "只用 CuTe view, 每线程 1 个输出",
      [&]() {
        sgemm_cute_view_naive_kernel<<<naive_grid, naive_block>>>(
            device_a, device_b, device_c, rows_m, cols_n, depth_k);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTolerance,
      kWarmupIters,
      kRepeatIters));

  results.push_back(benchmark_kernel(
      "L1_shared_tiled_f32",
      "shared memory + K-slicing 基线",
      [&]() {
        sgemm_cute_shared_tiled_kernel<kSmallBM, kSmallBN, kSmallBK>
            <<<small_grid, small_block_2d>>>(device_a,
                                             device_b,
                                             device_c,
                                             rows_m,
                                             cols_n,
                                             depth_k);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTolerance,
      kWarmupIters,
      kRepeatIters));

  results.push_back(benchmark_kernel(
      "L2_partition_copy_f32",
      "local_partition + copy_if",
      [&]() {
        sgemm_cute_partition_copy_kernel<kSmallBM,
                                         kSmallBN,
                                         kSmallBK,
                                         decltype(small_partition_a),
                                         decltype(small_partition_b)>
            <<<small_grid, small_partition_block>>>(device_a,
                                                    device_b,
                                                    device_c,
                                                    rows_m,
                                                    cols_n,
                                                    depth_k,
                                                    small_partition_a,
                                                    small_partition_b);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTolerance,
      kWarmupIters,
      kRepeatIters));

  results.push_back(benchmark_kernel(
      "L3_threadlayout_f32",
      "CuTe tutorial step1: thread layout + gemm",
      [&]() {
        sgemm_cute_threadlayout_kernel<kBigBM,
                                       kBigBN,
                                       kBigBK,
                                       decltype(thread_layout_a),
                                       decltype(thread_layout_b),
                                       decltype(thread_layout_c),
                                       decltype(smem_layout_a),
                                       decltype(smem_layout_b)><<<big_grid, big_block>>>(
            device_a,
            device_b,
            device_c,
            rows_m,
            cols_n,
            depth_k,
            thread_layout_a,
            thread_layout_b,
            thread_layout_c,
            smem_layout_a,
            smem_layout_b);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTolerance,
      kWarmupIters,
      kRepeatIters));

  if (can_run_vectorized_levels) {
    results.push_back(benchmark_kernel(
        "L4_tiledcopy_f32",
        "CuTe tutorial step2: TiledCopy + gemm",
        [&]() {
          sgemm_cute_tiledcopy_kernel<kBigBM,
                                      kBigBN,
                                      kBigBK,
                                      decltype(copy_a),
                                      decltype(copy_b),
                                      decltype(thread_layout_c),
                                      decltype(smem_layout_a),
                                      decltype(smem_layout_b)><<<big_grid, big_block>>>(
              device_a,
              device_b,
              device_c,
              rows_m,
              cols_n,
              depth_k,
              copy_a,
              copy_b,
              thread_layout_c,
              smem_layout_a,
              smem_layout_b);
        },
        device_c,
        host_c,
        host_reference,
        bytes_c,
        rows_m,
        cols_n,
        depth_k,
        kTolerance,
        kWarmupIters,
        kRepeatIters));

    results.push_back(benchmark_kernel(
        "L5_tiledmma_f32",
        "当前通用 CuTe 路线最强: TiledCopy + TiledMMA",
        [&]() {
          sgemm_cute_tiledmma_kernel<kBigBM,
                                     kBigBN,
                                     kBigBK,
                                     decltype(copy_a),
                                     decltype(copy_b),
                                     decltype(smem_layout_a),
                                     decltype(smem_layout_b)><<<big_grid, big_block>>>(
              device_a,
              device_b,
              device_c,
              rows_m,
              cols_n,
              depth_k,
              copy_a,
              copy_b,
              smem_layout_a,
              smem_layout_b);
        },
        device_c,
        host_c,
        host_reference,
        bytes_c,
        rows_m,
        cols_n,
        depth_k,
        kTolerance,
        kWarmupIters,
        kRepeatIters));
  }

  std::printf("%-22s | %-42s | %12s | %12s | %12s | %8s\n",
              "Kernel",
              "TeachingPoint",
              "MaxAbsErr",
              "Time(ms)",
              "TFLOPS",
              "Status");
  std::printf("-----------------------+--------------------------------------------+--------------+--------------+--------------+----------\n");

  bool all_passed = true;
  for (const BenchmarkResult &result : results) {
    const char *status_text = result.skipped ? "SKIP" : (result.passed ? "PASS" : "FAIL");
    std::printf("%-22s | %-42s | %12.6g | %12.4f | %12.4f | %8s\n",
                result.name,
                result.teaching_point,
                result.max_abs_error,
                result.avg_time_ms,
                result.tflops,
                status_text);
    all_passed = all_passed && (result.skipped || result.passed);
  }

  CHECK_CUDA(cudaFree(device_c));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_a));
  return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
