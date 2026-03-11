#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    cudaError_t error_code__ = (call);                                    \
    if (error_code__ != cudaSuccess) {                                    \
      std::fprintf(stderr, "CUDA error: %s @ %s:%d\n",                    \
                   cudaGetErrorString(error_code__), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

#define CHECK_CUBLAS(call)                                          \
  do {                                                              \
    cublasStatus_t status__ = (call);                               \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                        \
      std::fprintf(stderr, "cuBLAS error: %d @ %s:%d\n",            \
                   static_cast<int>(status__), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                      \
    }                                                               \
  } while (0)

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes) \
  asm volatile(                      \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

namespace wmma = nvcuda::wmma;

struct BenchmarkResult {
  const char *name = "";
  const char *teaching_point = "";
  float max_abs_error = 0.0f;
  float avg_time_ms = 0.0f;
  double tflops = 0.0;
  bool passed = false;
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
  std::printf("当前 demo 实现的是 float 输入 + TF32 WMMA + FP32 accumulate。\n");
  std::printf("建议编译时追加 NVCC_FLAGS=\"-arch=sm_80\" 或更高架构。\n");
}

void fill_matrix(std::vector<float> &matrix, const int rows, const int cols) {
  for (int row_index = 0; row_index < rows; ++row_index) {
    for (int col_index = 0; col_index < cols; ++col_index) {
      const int linear_index = row_index * cols + col_index;
      const float base_value = static_cast<float>((linear_index % 31) - 15) * 0.037f;
      const float row_bias = static_cast<float>((row_index % 7) - 3) * 0.0091f;
      const float col_bias = static_cast<float>((col_index % 9) - 4) * 0.0053f;
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

template <int BlockRowsM, int BlockDepthK, int SharedStrideA>
__device__ __forceinline__ void load_a_stage_async(float *shared_a_stage,
                                                   const float *matrix_a,
                                                   const int block_row_start,
                                                   const int block_k_start,
                                                   const int rows_m,
                                                   const int depth_k,
                                                   const int thread_index) {
  constexpr int kCopyWidth = 4;
  constexpr int kSegmentsPerRow = BlockDepthK / kCopyWidth;

  const int segment_index = thread_index;
  if (segment_index >= BlockRowsM * kSegmentsPerRow) {
    return;
  }

  const int tile_row = segment_index / kSegmentsPerRow;
  const int tile_col = (segment_index % kSegmentsPerRow) * kCopyWidth;
  const int source_row = block_row_start + tile_row;
  const int source_col = block_k_start + tile_col;

  float *shared_ptr = shared_a_stage + tile_row * SharedStrideA + tile_col;
  const float *global_ptr = matrix_a + source_row * depth_k + source_col;

  const bool in_bounds = source_row < rows_m && source_col + kCopyWidth <= depth_k;
  const bool aligned = (reinterpret_cast<std::uintptr_t>(global_ptr) & 0xF) == 0;
  if (in_bounds && aligned) {
    const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
    CP_ASYNC_CA(shared_addr, global_ptr, 16);
    return;
  }

#pragma unroll
  for (int offset = 0; offset < kCopyWidth; ++offset) {
    const int k_index = source_col + offset;
    shared_ptr[offset] =
        (source_row < rows_m && k_index < depth_k) ? matrix_a[source_row * depth_k + k_index]
                                                   : 0.0f;
  }
}

template <int BlockColsN, int BlockDepthK, int SharedStrideB, int ThreadsPerBlock>
__device__ __forceinline__ void load_b_stage_async(float *shared_b_stage,
                                                   const float *matrix_b,
                                                   const int block_col_start,
                                                   const int block_k_start,
                                                   const int cols_n,
                                                   const int depth_k,
                                                   const int thread_index) {
  constexpr int kCopyWidth = 4;
  constexpr int kSegmentsPerRow = BlockColsN / kCopyWidth;
  constexpr int kTotalSegments = BlockDepthK * kSegmentsPerRow;

  for (int segment_index = thread_index; segment_index < kTotalSegments;
       segment_index += ThreadsPerBlock) {
    const int tile_row = segment_index / kSegmentsPerRow;
    const int tile_col = (segment_index % kSegmentsPerRow) * kCopyWidth;
    const int source_row = block_k_start + tile_row;
    const int source_col = block_col_start + tile_col;

    float *shared_ptr = shared_b_stage + tile_row * SharedStrideB + tile_col;
    const float *global_ptr = matrix_b + source_row * cols_n + source_col;

    const bool in_bounds = source_row < depth_k && source_col + kCopyWidth <= cols_n;
    const bool aligned = (reinterpret_cast<std::uintptr_t>(global_ptr) & 0xF) == 0;
    if (in_bounds && aligned) {
      const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
      CP_ASYNC_CA(shared_addr, global_ptr, 16);
      continue;
    }

#pragma unroll
    for (int offset = 0; offset < kCopyWidth; ++offset) {
      const int n_index = source_col + offset;
      shared_ptr[offset] =
          (source_row < depth_k && n_index < cols_n) ? matrix_b[source_row * cols_n + n_index]
                                                     : 0.0f;
    }
  }
}

template <int BlockRowsM, int BlockDepthK, int SharedStrideA>
__device__ __forceinline__ void convert_a_stage_to_tf32(float *shared_a_stage,
                                                        const int thread_index) {
  constexpr int kCopyWidth = 4;
  constexpr int kSegmentsPerRow = BlockDepthK / kCopyWidth;

  const int segment_index = thread_index;
  if (segment_index >= BlockRowsM * kSegmentsPerRow) {
    return;
  }

  const int tile_row = segment_index / kSegmentsPerRow;
  const int tile_col = (segment_index % kSegmentsPerRow) * kCopyWidth;
  float *shared_ptr = shared_a_stage + tile_row * SharedStrideA + tile_col;

#pragma unroll
  for (int offset = 0; offset < kCopyWidth; ++offset) {
    shared_ptr[offset] = wmma::__float_to_tf32(shared_ptr[offset]);
  }
}

template <int BlockColsN, int BlockDepthK, int SharedStrideB, int ThreadsPerBlock>
__device__ __forceinline__ void convert_b_stage_to_tf32(float *shared_b_stage,
                                                        const int thread_index) {
  constexpr int kCopyWidth = 4;
  constexpr int kSegmentsPerRow = BlockColsN / kCopyWidth;
  constexpr int kTotalSegments = BlockDepthK * kSegmentsPerRow;

  for (int segment_index = thread_index; segment_index < kTotalSegments;
       segment_index += ThreadsPerBlock) {
    const int tile_row = segment_index / kSegmentsPerRow;
    const int tile_col = (segment_index % kSegmentsPerRow) * kCopyWidth;
    float *shared_ptr = shared_b_stage + tile_row * SharedStrideB + tile_col;

#pragma unroll
    for (int offset = 0; offset < kCopyWidth; ++offset) {
      shared_ptr[offset] = wmma::__float_to_tf32(shared_ptr[offset]);
    }
  }
}

template <int BlockRowsM = 64,
          int BlockColsN = 128,
          int BlockDepthK = 16,
          int WarpTilesM = 2,
          int WarpTilesN = 4,
          int WarpTileRowsM = 32,
          int WarpTileColsN = 32,
          int SharedPadA = 8,
          int SharedPadB = 8>
__global__ void sgemm_wmma_tf32_kernel(const float *matrix_a,
                                       const float *matrix_b,
                                       float *matrix_c,
                                       const int rows_m,
                                       const int cols_n,
                                       const int depth_k) {
  // WMMA 的 TF32 指令形状是 m16n16k8，因此 block 的 K 分块必须是 8 的整数倍。
  static_assert(BlockDepthK % 8 == 0, "WMMA tf32 requires K tile aligned to 8.");
  // 每个 warp 内部进一步拆成若干个 16x16 的 WMMA tile，所以 warp tile 的 M/N 也必须能被 16 整除。
  static_assert(WarpTileRowsM % 16 == 0 && WarpTileColsN % 16 == 0,
                "Warp tile must be composed of 16x16 WMMA tiles.");
  // 这里固定用 8 个 warp，也就是 256 个线程；整个 kernel 的映射关系就是围绕这个配置写死的。
  static_assert(WarpTilesM * WarpTilesN * 32 == 256,
                "This kernel is tuned for 8 warps / 256 threads per block.");

  // 单个 WMMA 指令在 M 维覆盖 16 行。
  constexpr int kWmmaM = 16;
  // 单个 WMMA 指令在 N 维覆盖 16 列。
  constexpr int kWmmaN = 16;
  // 单个 WMMA 指令在 K 维消费 8 个元素。
  constexpr int kWmmaK = 8;
  // block 内总线程数 = warp 数 * 32。
  constexpr int kThreadsPerBlock = WarpTilesM * WarpTilesN * 32;
  // 一个 warp 的输出块在 M 维会被拆成多少个 16x16 fragment。
  constexpr int kWarpFragmentsM = WarpTileRowsM / kWmmaM;
  // 一个 warp 的输出块在 N 维会被拆成多少个 16x16 fragment。
  constexpr int kWarpFragmentsN = WarpTileColsN / kWmmaN;
  // A tile 在 shared memory 里的行跨度；额外 pad 用来减轻 bank conflict。
  constexpr int kSharedStrideA = BlockDepthK + SharedPadA;
  // B tile 在 shared memory 里的行跨度；同样加入 pad。
  constexpr int kSharedStrideB = BlockColsN + SharedPadB;
  // 边界写回时，每个 warp 只需要一个 16x16 的 staging buffer。
  constexpr int kWarpStageStride = kWmmaM * kWmmaN;

  // block 内一维线性线程号；当前 block 配置是 256x1x1，所以直接取 threadIdx.x。
  const int thread_index = threadIdx.x;
  // 当前线程属于第几个 warp。
  const int warp_id = thread_index >> 5;
  // 当前线程在 warp 内的 lane id。
  const int lane_id = thread_index & 31;
  // warp 在 block tile 内负责的 warp-row 编号。
  const int warp_tile_row = warp_id / WarpTilesN;
  // warp 在 block tile 内负责的 warp-col 编号。
  const int warp_tile_col = warp_id % WarpTilesN;

  // 当前 block 在输出矩阵 C 中对应 tile 的起始行。
  const int block_row_start = blockIdx.y * BlockRowsM;
  // 当前 block 在输出矩阵 C 中对应 tile 的起始列。
  const int block_col_start = blockIdx.x * BlockColsN;

  // 动态 shared memory 原始首地址；后面手工切成 A/B/staging 三段。
  extern __shared__ __align__(32) unsigned char shared_raw[];
  // shared_a 存当前 block 的 A tile，布局是 [BlockRowsM, BlockDepthK + pad]。
  float *shared_a = reinterpret_cast<float *>(shared_raw);
  // shared_b 紧跟在 shared_a 后面，存当前 block 的 B tile。
  float *shared_b = shared_a + BlockRowsM * kSharedStrideA;
  // shared_stage 再往后，为边界 tile 写回准备一个 warp 级临时缓冲区。
  float *shared_stage = shared_b + BlockDepthK * kSharedStrideB;

  // 每个 warp 持有自己的寄存器累加器矩阵；元素类型是 float，表示 FP32 accumulate。
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
      accumulators[kWarpFragmentsM][kWarpFragmentsN];

  // 先把当前 warp 负责的所有 accumulator fragment 清零。
#pragma unroll
  for (int fragment_row = 0; fragment_row < kWarpFragmentsM; ++fragment_row) {
#pragma unroll
    for (int fragment_col = 0; fragment_col < kWarpFragmentsN; ++fragment_col) {
      // fill_fragment 会把整个 fragment 的寄存器值都置为 0.0f。
      wmma::fill_fragment(accumulators[fragment_row][fragment_col], 0.0f);
    }
  }

  // 沿着 K 维做 block 级分块；每次处理一个 [BlockRowsM, BlockColsN, BlockDepthK] 子问题。
  for (int block_k_start = 0; block_k_start < depth_k; block_k_start += BlockDepthK) {
    // 所有线程协作把 A 的 block tile 搬到 shared memory。
    for (int linear_index = thread_index; linear_index < BlockRowsM * BlockDepthK;
         linear_index += kThreadsPerBlock) {
      // 把一维 linear index 映射回 tile 内二维坐标 (row, col)。
      const int tile_row = linear_index / BlockDepthK;
      const int tile_col = linear_index % BlockDepthK;
      // A 源矩阵中的实际行号 = 当前 block 的起始行 + tile 内局部行号。
      const int source_row = block_row_start + tile_row;
      // A 源矩阵中的实际列号 = 当前 K tile 的起始列 + tile 内局部列号。
      const int source_col = block_k_start + tile_col;
      // 有效范围内读取 A，并在写入 shared 时先量化成 TF32；越界元素补 0。
      shared_a[tile_row * kSharedStrideA + tile_col] =
          (source_row < rows_m && source_col < depth_k)
              ? wmma::__float_to_tf32(matrix_a[source_row * depth_k + source_col])
              : 0.0f;
    }

    // 所有线程协作把 B 的 block tile 搬到 shared memory。
    for (int linear_index = thread_index; linear_index < BlockDepthK * BlockColsN;
         linear_index += kThreadsPerBlock) {
      // 把一维 linear index 映射回 B tile 内二维坐标。
      const int tile_row = linear_index / BlockColsN;
      const int tile_col = linear_index % BlockColsN;
      // B 源矩阵中的实际行号 = 当前 K tile 的起始行 + tile 内局部行号。
      const int source_row = block_k_start + tile_row;
      // B 源矩阵中的实际列号 = 当前 block 的起始列 + tile 内局部列号。
      const int source_col = block_col_start + tile_col;
      // 同样先转 TF32，越界则补 0。
      shared_b[tile_row * kSharedStrideB + tile_col] =
          (source_row < depth_k && source_col < cols_n)
              ? wmma::__float_to_tf32(matrix_b[source_row * cols_n + source_col])
              : 0.0f;
    }

    // 等待整个 block 的 A/B tile 都写入 shared memory 完成。
    __syncthreads();

    // 当前 block K tile 还要继续拆成多个 WMMA k=8 的 micro-tile。
    for (int tile_k_offset = 0; tile_k_offset < BlockDepthK; tile_k_offset += kWmmaK) {
      // fragment_a 保存当前 warp 需要参与一次 mma_sync 的 A 子块。
      wmma::fragment<wmma::matrix_a,
                     kWmmaM,
                     kWmmaN,
                     kWmmaK,
                     wmma::precision::tf32,
                     wmma::row_major>
          fragment_a[kWarpFragmentsM];
      // fragment_b 保存当前 warp 需要参与一次 mma_sync 的 B 子块。
      wmma::fragment<wmma::matrix_b,
                     kWmmaM,
                     kWmmaN,
                     kWmmaK,
                     wmma::precision::tf32,
                     wmma::row_major>
          fragment_b[kWarpFragmentsN];

      // 逐个加载本 warp 在 M 维负责的所有 A fragments。
#pragma unroll
      for (int fragment_row = 0; fragment_row < kWarpFragmentsM; ++fragment_row) {
        // 当前 A fragment 在 warp tile 内的起始行。
        const int a_row = warp_tile_row * WarpTileRowsM + fragment_row * kWmmaM;
        // 从 shared_a 载入一个 16x8 的 TF32 fragment。
        wmma::load_matrix_sync(fragment_a[fragment_row],
                               shared_a + a_row * kSharedStrideA + tile_k_offset,
                               kSharedStrideA);
      }

      // 逐个加载本 warp 在 N 维负责的所有 B fragments。
#pragma unroll
      for (int fragment_col = 0; fragment_col < kWarpFragmentsN; ++fragment_col) {
        // 当前 B fragment 在 warp tile 内的起始列。
        const int b_col = warp_tile_col * WarpTileColsN + fragment_col * kWmmaN;
        // 从 shared_b 载入一个 8x16 的 TF32 fragment。
        wmma::load_matrix_sync(fragment_b[fragment_col],
                               shared_b + tile_k_offset * kSharedStrideB + b_col,
                               kSharedStrideB);
      }

      // 两层小循环做的就是 warp 负责的所有 C 子块的 outer-product 更新。
#pragma unroll
      for (int fragment_row = 0; fragment_row < kWarpFragmentsM; ++fragment_row) {
#pragma unroll
        for (int fragment_col = 0; fragment_col < kWarpFragmentsN; ++fragment_col) {
          // D = A * B + C；这里直接把结果累加回原 accumulator。
          wmma::mma_sync(accumulators[fragment_row][fragment_col],
                         fragment_a[fragment_row],
                         fragment_b[fragment_col],
                         accumulators[fragment_row][fragment_col]);
        }
      }
    }

    // 当前 K tile 计算结束，再同步一次，确保下一轮覆盖 shared_a/shared_b 时没有线程还在读。
    __syncthreads();
  }

  // 每个 warp 拿到自己专属的 16x16 staging buffer，用于边界回写。
  float *warp_stage = shared_stage + warp_id * kWarpStageStride;
  // WMMA 直接写 row-major global memory 时，ldm 需要满足对齐要求；这里要求 N 能被 4 整除。
  const bool can_store_direct = (cols_n % 4) == 0;

  // 开始把 warp 内所有 accumulator fragments 写回到 C。
#pragma unroll
  for (int fragment_row = 0; fragment_row < kWarpFragmentsM; ++fragment_row) {
#pragma unroll
    for (int fragment_col = 0; fragment_col < kWarpFragmentsN; ++fragment_col) {
      // 当前 fragment 对应到全局 C 中的起始行。
      const int global_row =
          block_row_start + warp_tile_row * WarpTileRowsM + fragment_row * kWmmaM;
      // 当前 fragment 对应到全局 C 中的起始列。
      const int global_col =
          block_col_start + warp_tile_col * WarpTileColsN + fragment_col * kWmmaN;

      // 如果这个 16x16 tile 完全落在有效区域内，且行跨度满足 WMMA store 的要求，就直接写回 global。
      if (can_store_direct && global_row + kWmmaM <= rows_m &&
          global_col + kWmmaN <= cols_n) {
        // 直接把 fragment 以 row-major 形式存入 C。
        wmma::store_matrix_sync(matrix_c + global_row * cols_n + global_col,
                                accumulators[fragment_row][fragment_col],
                                cols_n,
                                wmma::mem_row_major);
        // 完整 tile 处理完就进入下一个 fragment。
        continue;
      }

      // 边界 tile 不能直接存到 global，因此先存到每个 warp 自己的 shared staging buffer。
      wmma::store_matrix_sync(
          warp_stage, accumulators[fragment_row][fragment_col], kWmmaN, wmma::mem_row_major);

      // warp 内同步，确保 staging buffer 中 16x16 数据完整可见。
      __syncwarp();

      // 32 个 lane 分工把 staging buffer 中的 256 个元素搬回 global C。
      for (int linear_index = lane_id; linear_index < kWarpStageStride; linear_index += 32) {
        // 把一维下标还原成 staging tile 内的局部行。
        const int local_row = linear_index / kWmmaN;
        // 把一维下标还原成 staging tile 内的局部列。
        const int local_col = linear_index % kWmmaN;
        // 只有全局索引仍然落在有效矩阵范围内时才写回。
        if (global_row + local_row < rows_m && global_col + local_col < cols_n) {
          // staging -> global 的标量回写路径。
          matrix_c[(global_row + local_row) * cols_n + (global_col + local_col)] =
              warp_stage[linear_index];
        }
      }

      // 再同步一次，避免某些 lane 还没读完 warp_stage 就被下一个 fragment 覆盖。
      __syncwarp();
    }
  }
}

template <int BlockRowsM = 64,
          int BlockColsN = 128,
          int BlockDepthK = 16,
          int WarpTilesM = 2,
          int WarpTilesN = 4,
          int WarpTileRowsM = 32,
          int WarpTileColsN = 32,
          int SharedPadA = 8,
          int SharedPadB = 8>
__global__ void sgemm_wmma_tf32_cp_async_dbuf_kernel(const float *matrix_a,
                                                     const float *matrix_b,
                                                     float *matrix_c,
                                                     const int rows_m,
                                                     const int cols_n,
                                                     const int depth_k) {
  static_assert(BlockDepthK % 8 == 0, "WMMA tf32 requires K tile aligned to 8.");
  static_assert(BlockDepthK % 4 == 0, "cp.async vector copy expects BK aligned to 4.");
  static_assert(BlockColsN % 4 == 0, "cp.async vector copy expects BN aligned to 4.");
  static_assert(WarpTileRowsM % 16 == 0 && WarpTileColsN % 16 == 0,
                "Warp tile must be composed of 16x16 WMMA tiles.");
  static_assert(WarpTilesM * WarpTilesN * 32 == 256,
                "This kernel is tuned for 8 warps / 256 threads per block.");

  constexpr int kStages = 2;
  constexpr int kWmmaM = 16;
  constexpr int kWmmaN = 16;
  constexpr int kWmmaK = 8;
  constexpr int kThreadsPerBlock = WarpTilesM * WarpTilesN * 32;
  constexpr int kWarpFragmentsM = WarpTileRowsM / kWmmaM;
  constexpr int kWarpFragmentsN = WarpTileColsN / kWmmaN;
  constexpr int kSharedStrideA = BlockDepthK + SharedPadA;
  constexpr int kSharedStrideB = BlockColsN + SharedPadB;
  constexpr int kWarpStageStride = kWmmaM * kWmmaN;
  constexpr int kATileElements = BlockRowsM * kSharedStrideA;
  constexpr int kBTileElements = BlockDepthK * kSharedStrideB;

  const int thread_index = threadIdx.x;
  const int warp_id = thread_index >> 5;
  const int lane_id = thread_index & 31;
  const int warp_tile_row = warp_id / WarpTilesN;
  const int warp_tile_col = warp_id % WarpTilesN;

  const int block_row_start = blockIdx.y * BlockRowsM;
  const int block_col_start = blockIdx.x * BlockColsN;
  const int num_k_tiles = (depth_k + BlockDepthK - 1) / BlockDepthK;

  extern __shared__ __align__(32) unsigned char shared_raw[];
  float *shared_a = reinterpret_cast<float *>(shared_raw);
  float *shared_b = shared_a + kStages * kATileElements;
  float *shared_stage = shared_b + kStages * kBTileElements;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
      accumulators[kWarpFragmentsM][kWarpFragmentsN];

#pragma unroll
  for (int fragment_row = 0; fragment_row < kWarpFragmentsM; ++fragment_row) {
#pragma unroll
    for (int fragment_col = 0; fragment_col < kWarpFragmentsN; ++fragment_col) {
      wmma::fill_fragment(accumulators[fragment_row][fragment_col], 0.0f);
    }
  }

  const int preload_tiles = num_k_tiles > kStages ? kStages : num_k_tiles;
  for (int tile_index = 0; tile_index < preload_tiles; ++tile_index) {
    float *shared_a_stage = shared_a + (tile_index % kStages) * kATileElements;
    float *shared_b_stage = shared_b + (tile_index % kStages) * kBTileElements;
    const int block_k_start = tile_index * BlockDepthK;

    load_a_stage_async<BlockRowsM, BlockDepthK, kSharedStrideA>(shared_a_stage,
                                                                matrix_a,
                                                                block_row_start,
                                                                block_k_start,
                                                                rows_m,
                                                                depth_k,
                                                                thread_index);
    load_b_stage_async<BlockColsN, BlockDepthK, kSharedStrideB, kThreadsPerBlock>(
        shared_b_stage,
        matrix_b,
        block_col_start,
        block_k_start,
        cols_n,
        depth_k,
        thread_index);
    CP_ASYNC_COMMIT_GROUP();
  }

  for (int tile_index = 0; tile_index < num_k_tiles; ++tile_index) {
    const int current_stage = tile_index & 1;
    float *shared_a_stage = shared_a + current_stage * kATileElements;
    float *shared_b_stage = shared_b + current_stage * kBTileElements;

    if (tile_index + 1 < num_k_tiles) {
      CP_ASYNC_WAIT_GROUP(1);
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    convert_a_stage_to_tf32<BlockRowsM, BlockDepthK, kSharedStrideA>(shared_a_stage,
                                                                     thread_index);
    convert_b_stage_to_tf32<BlockColsN, BlockDepthK, kSharedStrideB, kThreadsPerBlock>(
        shared_b_stage, thread_index);
    __syncthreads();

    for (int tile_k_offset = 0; tile_k_offset < BlockDepthK; tile_k_offset += kWmmaK) {
      wmma::fragment<wmma::matrix_a,
                     kWmmaM,
                     kWmmaN,
                     kWmmaK,
                     wmma::precision::tf32,
                     wmma::row_major>
          fragment_a[kWarpFragmentsM];
      wmma::fragment<wmma::matrix_b,
                     kWmmaM,
                     kWmmaN,
                     kWmmaK,
                     wmma::precision::tf32,
                     wmma::row_major>
          fragment_b[kWarpFragmentsN];

#pragma unroll
      for (int fragment_row = 0; fragment_row < kWarpFragmentsM; ++fragment_row) {
        const int a_row = warp_tile_row * WarpTileRowsM + fragment_row * kWmmaM;
        wmma::load_matrix_sync(fragment_a[fragment_row],
                               shared_a_stage + a_row * kSharedStrideA + tile_k_offset,
                               kSharedStrideA);
      }

#pragma unroll
      for (int fragment_col = 0; fragment_col < kWarpFragmentsN; ++fragment_col) {
        const int b_col = warp_tile_col * WarpTileColsN + fragment_col * kWmmaN;
        wmma::load_matrix_sync(fragment_b[fragment_col],
                               shared_b_stage + tile_k_offset * kSharedStrideB + b_col,
                               kSharedStrideB);
      }

#pragma unroll
      for (int fragment_row = 0; fragment_row < kWarpFragmentsM; ++fragment_row) {
#pragma unroll
        for (int fragment_col = 0; fragment_col < kWarpFragmentsN; ++fragment_col) {
          wmma::mma_sync(accumulators[fragment_row][fragment_col],
                         fragment_a[fragment_row],
                         fragment_b[fragment_col],
                         accumulators[fragment_row][fragment_col]);
        }
      }
    }

    const int next_tile = tile_index + kStages;
    if (next_tile < num_k_tiles) {
      float *shared_a_next = shared_a + current_stage * kATileElements;
      float *shared_b_next = shared_b + current_stage * kBTileElements;
      const int block_k_start = next_tile * BlockDepthK;

      load_a_stage_async<BlockRowsM, BlockDepthK, kSharedStrideA>(shared_a_next,
                                                                  matrix_a,
                                                                  block_row_start,
                                                                  block_k_start,
                                                                  rows_m,
                                                                  depth_k,
                                                                  thread_index);
      load_b_stage_async<BlockColsN, BlockDepthK, kSharedStrideB, kThreadsPerBlock>(
          shared_b_next,
          matrix_b,
          block_col_start,
          block_k_start,
          cols_n,
          depth_k,
          thread_index);
      CP_ASYNC_COMMIT_GROUP();
    }

    __syncthreads();
  }

  float *warp_stage = shared_stage + warp_id * kWarpStageStride;
  const bool can_store_direct = (cols_n % 4) == 0;

#pragma unroll
  for (int fragment_row = 0; fragment_row < kWarpFragmentsM; ++fragment_row) {
#pragma unroll
    for (int fragment_col = 0; fragment_col < kWarpFragmentsN; ++fragment_col) {
      const int global_row =
          block_row_start + warp_tile_row * WarpTileRowsM + fragment_row * kWmmaM;
      const int global_col =
          block_col_start + warp_tile_col * WarpTileColsN + fragment_col * kWmmaN;

      if (can_store_direct && global_row + kWmmaM <= rows_m &&
          global_col + kWmmaN <= cols_n) {
        wmma::store_matrix_sync(matrix_c + global_row * cols_n + global_col,
                                accumulators[fragment_row][fragment_col],
                                cols_n,
                                wmma::mem_row_major);
        continue;
      }

      wmma::store_matrix_sync(
          warp_stage, accumulators[fragment_row][fragment_col], kWmmaN, wmma::mem_row_major);
      __syncwarp();

      for (int linear_index = lane_id; linear_index < kWarpStageStride; linear_index += 32) {
        const int local_row = linear_index / kWmmaN;
        const int local_col = linear_index % kWmmaN;
        if (global_row + local_row < rows_m && global_col + local_col < cols_n) {
          matrix_c[(global_row + local_row) * cols_n + (global_col + local_col)] =
              warp_stage[linear_index];
        }
      }

      __syncwarp();
    }
  }
}

void cublas_row_major_sgemm(cublasHandle_t handle,
                            const float *matrix_a,
                            const float *matrix_b,
                            float *matrix_c,
                            const int rows_m,
                            const int cols_n,
                            const int depth_k,
                            const cublasComputeType_t compute_type,
                            const cublasGemmAlgo_t algorithm) {
  static const float alpha = 1.0f;
  static const float beta = 0.0f;

  CHECK_CUBLAS(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            cols_n,
                            rows_m,
                            depth_k,
                            &alpha,
                            matrix_b,
                            CUDA_R_32F,
                            cols_n,
                            matrix_a,
                            CUDA_R_32F,
                            depth_k,
                            &beta,
                            matrix_c,
                            CUDA_R_32F,
                            cols_n,
                            compute_type,
                            algorithm));
}

int main(int argc, char **argv) {
  constexpr int kDefaultSize = 512;
  constexpr int kWarmupIters = 5;
  constexpr int kRepeatIters = 20;
  constexpr float kFp32Tolerance = 1e-3f;
  constexpr float kTf32Tolerance = 2e-1f;

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

  cudaDeviceProp device_prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&device_prop, 0));
  if (device_prop.major < 8) {
    std::fprintf(stderr,
                 "当前 GPU 为 sm_%d%d, TF32 WMMA 至少需要 sm_80。\n",
                 device_prop.major,
                 device_prop.minor);
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

  constexpr int kBlockRowsM = 64;
  constexpr int kBlockColsN = 128;
  constexpr int kBlockDepthK = 16;
  constexpr int kWarpTilesM = 2;
  constexpr int kWarpTilesN = 4;
  constexpr int kWarpTileRowsM = 32;
  constexpr int kWarpTileColsN = 32;
  constexpr int kSharedPadA = 8;
  constexpr int kSharedPadB = 8;
  constexpr int kThreadsPerBlock = 256;
  constexpr int kSharedStrideA = kBlockDepthK + kSharedPadA;
  constexpr int kSharedStrideB = kBlockColsN + kSharedPadB;
  constexpr size_t kSharedBytesBaseline =
      static_cast<size_t>(kBlockRowsM) * kSharedStrideA * sizeof(float) +
      static_cast<size_t>(kBlockDepthK) * kSharedStrideB * sizeof(float) +
      static_cast<size_t>(kWarpTilesM * kWarpTilesN * 16 * 16) *
          sizeof(float);
  constexpr size_t kSharedBytesAsync =
      2 * static_cast<size_t>(kBlockRowsM) * kSharedStrideA * sizeof(float) +
      2 * static_cast<size_t>(kBlockDepthK) * kSharedStrideB * sizeof(float) +
      static_cast<size_t>(kWarpTilesM * kWarpTilesN * 16 * 16) *
          sizeof(float);

  const dim3 wmma_block(kThreadsPerBlock);
  const dim3 wmma_grid((cols_n + kBlockColsN - 1) / kBlockColsN,
                       (rows_m + kBlockRowsM - 1) / kBlockRowsM);

  cublasHandle_t cublas_handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&cublas_handle));

  std::printf("WMMA TF32 SGEMM demo\n");
  std::printf("GPU: %s (sm_%d%d)\n", device_prop.name, device_prop.major, device_prop.minor);
  std::printf("矩阵尺寸: M=%d, N=%d, K=%d\n", rows_m, cols_n, depth_k);
  std::printf("Block tile: %dx%dx%d, warp tile: %dx%d\n\n",
              kBlockRowsM,
              kBlockColsN,
              kBlockDepthK,
              kWarpTileRowsM,
              kWarpTileColsN);

  std::vector<BenchmarkResult> results;
  results.push_back(benchmark_kernel(
      "wmma_tf32_baseline",
      "shared tiling + 8warp/block + TF32 tensor core",
      [&]() {
        sgemm_wmma_tf32_kernel<kBlockRowsM,
                               kBlockColsN,
                               kBlockDepthK,
                               kWarpTilesM,
                               kWarpTilesN,
                               kWarpTileRowsM,
                               kWarpTileColsN,
                               kSharedPadA,
                               kSharedPadB><<<wmma_grid, wmma_block, kSharedBytesBaseline>>>(
            device_a, device_b, device_c, rows_m, cols_n, depth_k);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTf32Tolerance,
      kWarmupIters,
      kRepeatIters));

  results.push_back(benchmark_kernel(
      "wmma_tf32_cp_async",
      "cp.async + double buffer + TF32 tensor core",
      [&]() {
        sgemm_wmma_tf32_cp_async_dbuf_kernel<kBlockRowsM,
                                             kBlockColsN,
                                             kBlockDepthK,
                                             kWarpTilesM,
                                             kWarpTilesN,
                                             kWarpTileRowsM,
                                             kWarpTileColsN,
                                             kSharedPadA,
                                             kSharedPadB>
            <<<wmma_grid, wmma_block, kSharedBytesAsync>>>(
                device_a, device_b, device_c, rows_m, cols_n, depth_k);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTf32Tolerance,
      kWarmupIters,
      kRepeatIters));

  results.push_back(benchmark_kernel(
      "cublas_fp32",
      "pedantic fp32 baseline",
      [&]() {
        cublas_row_major_sgemm(cublas_handle,
                               device_a,
                               device_b,
                               device_c,
                               rows_m,
                               cols_n,
                               depth_k,
                               CUBLAS_COMPUTE_32F_PEDANTIC,
                               CUBLAS_GEMM_DEFAULT);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kFp32Tolerance,
      kWarmupIters,
      kRepeatIters));

  results.push_back(benchmark_kernel(
      "cublas_tf32",
      "official tf32 baseline",
      [&]() {
        cublas_row_major_sgemm(cublas_handle,
                               device_a,
                               device_b,
                               device_c,
                               rows_m,
                               cols_n,
                               depth_k,
                               CUBLAS_COMPUTE_32F_FAST_TF32,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      },
      device_c,
      host_c,
      host_reference,
      bytes_c,
      rows_m,
      cols_n,
      depth_k,
      kTf32Tolerance,
      kWarmupIters,
      kRepeatIters));

  std::printf("%-24s | %-38s | %12s | %12s | %12s | %8s\n",
              "Kernel",
              "TeachingPoint",
              "MaxAbsErr",
              "Time(ms)",
              "TFLOPS",
              "Status");
  std::printf(
      "-------------------------+----------------------------------------+--------------+--------------+--------------+----------\n");

  for (const BenchmarkResult &result : results) {
    std::printf("%-24s | %-38s | %12.6g | %12.4f | %12.4f | %8s\n",
                result.name,
                result.teaching_point,
                result.max_abs_error,
                result.avg_time_ms,
                result.tflops,
                result.passed ? "PASS" : "FAIL");
  }

  CHECK_CUBLAS(cublasDestroy(cublas_handle));
  CHECK_CUDA(cudaFree(device_c));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_a));
  return EXIT_SUCCESS;
}
