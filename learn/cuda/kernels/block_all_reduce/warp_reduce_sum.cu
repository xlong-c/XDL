#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status = call;                                           \
    if (status != cudaSuccess) {                                         \
      printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,           \
             cudaGetErrorString(status));                                \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = (WARP_SIZE >> 1); offset >= 1; offset >>= 1) {
    val = val + __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Block-level reduction using shared memory and warp reduction
__device__ __forceinline__ float block_reduce_sum(float val) {
  static __shared__ float shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warp_reduce_sum(val);

  if (lane == 0) {
    shared[wid] = val;
  }

  __syncthreads();

  // Read from shared memory only if that warp existed
  val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;

  if (wid == 0) {
    val = warp_reduce_sum(val);
  }

  return val;
}

// 1. Total Sum Reduction
__global__ void reduce_total_kernel(const float* input, float* output, int size) {
  float sum = 0.0f;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < size; i += stride) {
    sum += input[i];
  }

  sum = block_reduce_sum(sum);

  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

// 2. Row Sum Reduction (M rows, N cols)
// Each block handles one or more rows. For simplicity, each block handles one row if N is large.
__global__ void reduce_rows_kernel(const float* input, float* output, int M, int N) {
  int row = blockIdx.x;
  if (row >= M) return;

  float sum = 0.0f;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    sum += input[row * N + col];
  }

  sum = block_reduce_sum(sum);

  if (threadIdx.x == 0) {
    output[row] = sum;
  }
}

// 3. Column Sum Reduction (M rows, N cols)
// To keep memory access coalesced, threads in a block process contiguous elements (different columns).
__global__ void reduce_cols_kernel(const float* input, float* output, int M, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) return;

  float sum = 0.0f;
  for (int row = 0; row < M; ++row) {
    sum += input[row * N + col];
  }

  output[col] = sum;
}

// Optimized Column Sum Reduction using shared memory for better performance if M is large
// but the naive one is already coalesced for reading input[row * N + col] when N is large.
// Let's stick to the coalesced one and maybe add a more tiled version if needed.

int main() {
  const int M = 1024;
  const int N = 2048;
  const int size = M * N;

  std::vector<float> h_input(size);
  for (int i = 0; i < size; ++i) h_input[i] = 1.0f; // Initialize with 1.0

  float *d_input, *d_output_total, *d_output_rows, *d_output_cols;
  CHECK_CUDA(cudaMalloc(&d_input, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output_total, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output_rows, M * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output_cols, N * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_output_total, 0, sizeof(float)));

  // 1. Total Sum
  cudaEvent_t start, stop;
  float milliseconds = 0;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (grid_size > 1024) grid_size = 1024;
  reduce_total_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_output_total, size);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Total Sum Time: %f ms\n", milliseconds);

  // 2. Row Sum
  CHECK_CUDA(cudaEventRecord(start));
  reduce_rows_kernel<<<M, BLOCK_SIZE>>>(d_input, d_output_rows, M, N);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Row Sum Time: %f ms\n", milliseconds);

  // 3. Column Sum
  CHECK_CUDA(cudaEventRecord(start));
  int col_grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  reduce_cols_kernel<<<col_grid, BLOCK_SIZE>>>(d_input, d_output_cols, M, N);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Col Sum Time: %f ms\n", milliseconds);

  // Verify Total Sum
  float h_output_total;
  CHECK_CUDA(cudaMemcpy(&h_output_total, d_output_total, sizeof(float), cudaMemcpyDeviceToHost));
  printf("Total Sum: Expected %f, Got %f\n", (float)size, h_output_total);

  // Verify Row Sum (first row)
  std::vector<float> h_output_rows(M);
  CHECK_CUDA(cudaMemcpy(h_output_rows.data(), d_output_rows, M * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Row 0 Sum: Expected %f, Got %f\n", (float)N, h_output_rows[0]);

  // Verify Column Sum (first column)
  std::vector<float> h_output_cols(N);
  CHECK_CUDA(cudaMemcpy(h_output_cols.data(), d_output_cols, N * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Col 0 Sum: Expected %f, Got %f\n", (float)M, h_output_cols[0]);

  cudaFree(d_input);
  cudaFree(d_output_total);
  cudaFree(d_output_rows);
  cudaFree(d_output_cols);

  return 0;
}