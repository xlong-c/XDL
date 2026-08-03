/**
 * CUDA Block and Warp Reduce Demo
 *
 * Demonstrates:
 * 1. Warp-level reduction using warp shuffle intrinsics (__shfl_down_sync)
 * 2. Block-level reduction using shared memory and warp reduce
 * 3. Performance comparison: naive approach vs optimized warp/block reduce
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Warp size constant
#define WARP_SIZE 32

// ============================================================================
// 1. WARP-LEVEL REDUCTION (using shuffle intrinsics)
// ============================================================================

/**
 * Warp reduce sum using __shfl_down_sync
 * Each thread in warp contributes its value, result available in lane 0
 */
__inline__ __device__ float warpReduceSum(float val) {
// Use __shfl_down_sync to shift values down in the warp
// The mask 0xffffffff means all threads in warp participate
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

/**
 * Warp reduce max using __shfl_down_sync
 */
__inline__ __device__ float warpReduceMax(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

/**
 * Kernel: Demonstrate warp-level reduction
 * Each warp reduces its own 32-element portion
 */
__global__ void warpReduceDemoKernel(const float *input, float *output, int n) {
  int tid = threadIdx.x;
  int warpId = tid / WARP_SIZE; // Which warp within block
  int laneId = tid % WARP_SIZE; // Which thread within warp

  // Each warp processes one 32-element chunk
  int dataIdx = blockIdx.x * blockDim.x + tid;

  float val = 0.0f;
  if (dataIdx < n) {
    val = input[dataIdx];
  }

  // Perform warp reduction
  float warpSum = warpReduceSum(val);

  // Only lane 0 has the complete sum
  if (laneId == 0) {
    // Store result (one per warp)
    int outputIdx = blockIdx.x * (blockDim.x / WARP_SIZE) + warpId;
    output[outputIdx] = warpSum;
  }
}

// ============================================================================
// 2. BLOCK-LEVEL REDUCTION (using shared memory + warp reduce)
// ============================================================================

/**
 * Block reduce sum using two-level approach:
 * 1. Each warp reduces its portion using warp shuffle
 * 2. Lane 0 from each warp writes to shared memory
 * 3. First warp reduces the warp sums
 * 4. Final result in thread 0
 */
template <int BLOCK_SIZE>
__inline__ __device__ float blockReduceSum(float val) {
  // Shared memory for warp-level results
  // Need BLOCK_SIZE / WARP_SIZE entries
  __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

  int tid = threadIdx.x;
  int laneId = tid % WARP_SIZE;
  int warpId = tid / WARP_SIZE;

  // Step 1: Warp-level reduction
  val = warpReduceSum(val);

  // Step 2: Write warp sum to shared memory
  if (laneId == 0) {
    shared[warpId] = val;
  }
  __syncthreads();

  // Step 3: First warp reduces all warp sums
  if (tid < WARP_SIZE) {
    // Read from shared memory (or 0 if beyond number of warps)
    val = (tid < BLOCK_SIZE / WARP_SIZE) ? shared[tid] : 0.0f;
    val = warpReduceSum(val);
  }

  return val;
}

/**
 * Kernel: Demonstrate block-level reduction
 * Each block reduces its entire portion to a single sum
 */
template <int BLOCK_SIZE>
__global__ void blockReduceDemoKernel(const float *input, float *output,
                                      int n) {
  int tid = threadIdx.x;
  int globalIdx = blockIdx.x * BLOCK_SIZE + tid;

  // Each thread loads one element (or 0 if out of bounds)
  float val = 0.0f;
  if (globalIdx < n) {
    val = input[globalIdx];
  }

  // Perform block-level reduction
  float blockSum = blockReduceSum<BLOCK_SIZE>(val);

  // Only thread 0 writes the result
  if (tid == 0) {
    output[blockIdx.x] = blockSum;
  }
}

// ============================================================================
// 3. NAIVE REDUCTION (for comparison)
// ============================================================================

/**
 * Naive reduction: Each thread adds to global memory with atomicAdd
 * Very slow due to contention on single memory location
 */
__global__ void naiveReduceKernel(const float *input, float *output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    atomicAdd(output, input[tid]);
  }
}

/**
 * Sequential reduction on CPU (for verification)
 */
double cpuReduceSum(const float *data, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += (double)data[i];
  }
  return sum;
}

// ============================================================================
// 4. MAIN FUNCTION AND TIMING
// ============================================================================

int main() {
  // Problem size
  int n = 1 << 20; // 1M elements
  size_t bytes = n * sizeof(float);

  printf("CUDA Block and Warp Reduce Demo\n");
  printf("================================\n");
  printf("Array size: %d elements (%.2f MB)\n\n", n, bytes / (1024.0 * 1024.0));

  // Allocate host memory
  float *h_input = (float *)malloc(bytes);
  float h_output = 0.0f;

  // Initialize data: values 1.0f + random(-1.0f, 1.0f)
  srand((unsigned int)time(NULL));
  for (int i = 0; i < n; i++) {
    float noise = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    h_input[i] = 1.0f + noise;
  }
  double expected = cpuReduceSum(h_input, n);
  printf("Expected sum (CPU): %.6f\n\n", expected);

  // Allocate device memory
  float *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

  // Setup timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float ms;

  // ============================================================================
  // Test 1: Naive Atomic Reduction (slow baseline)
  // ============================================================================
  printf("Test 1: Naive Atomic Reduction\n");
  printf("-------------------------------\n");

  CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  CUDA_CHECK(cudaEventRecord(start));
  naiveReduceKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(
      cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

  printf("Result: %.6f\n", h_output);
  printf("Error:  %.6f\n", fabs((double)h_output - expected));
  printf("Time:   %.3f ms\n\n", ms);

  // ============================================================================
  // Test 2: Warp-Level Reduction (intermediate)
  // ============================================================================
  printf("Test 2: Warp-Level Reduction (Shuffle)\n");
  printf("----------------------------------------\n");

  // For warp reduce, each warp outputs one value
  int numWarpsPerBlock = blockSize / WARP_SIZE;
  int outputSize = gridSize * numWarpsPerBlock;
  float *d_warpOutput;
  CUDA_CHECK(cudaMalloc(&d_warpOutput, outputSize * sizeof(float)));

  CUDA_CHECK(cudaEventRecord(start));
  warpReduceDemoKernel<<<gridSize, blockSize>>>(d_input, d_warpOutput, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Host-side reduction of warp outputs
  float *h_warpOutput = (float *)malloc(outputSize * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_warpOutput, d_warpOutput, outputSize * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double h_warpSum = 0.0;
  for (int i = 0; i < outputSize; i++) {
    h_warpSum += (double)h_warpOutput[i];
  }

  printf("Result: %.6f\n", h_warpSum);
  printf("Error:  %.6f\n", fabs(h_warpSum - expected));
  printf("Time:   %.3f ms (plus host reduction)\n\n", ms);

  free(h_warpOutput);
  CUDA_CHECK(cudaFree(d_warpOutput));

  // ============================================================================
  // Test 3: Block-Level Reduction (full tree reduction)
  // ============================================================================
  printf("Test 3: Block-Level Reduction (Tree)\n");
  printf("------------------------------------\n");

  // Reallocate output for block-level results
  float *d_blockOutput;
  CUDA_CHECK(cudaMalloc(&d_blockOutput, gridSize * sizeof(float)));

  CUDA_CHECK(cudaEventRecord(start));
  blockReduceDemoKernel<256><<<gridSize, 256>>>(d_input, d_blockOutput, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Host-side reduction of block outputs
  float *h_blockOutput = (float *)malloc(gridSize * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_blockOutput, d_blockOutput, gridSize * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double h_blockSum = 0.0;
  for (int i = 0; i < gridSize; i++) {
    h_blockSum += (double)h_blockOutput[i];
  }
  printf("Result: %.6f\n", h_blockSum);
  printf("Error:  %.6f\n", fabs(h_blockSum - expected));
  printf("Time:   %.3f ms (plus host reduction)\n\n", ms);

  free(h_blockOutput);
  CUDA_CHECK(cudaFree(d_blockOutput));

  // ============================================================================
  // Summary
  // ============================================================================
  printf("Summary\n");
  printf("=======\n");
  printf(
      "Block and Warp Reduce techniques compared to naive atomic approach.\n");
  printf("Key optimizations:\n");
  printf("  - Warp shuffle: O(log n) steps within warp, no shared memory\n");
  printf("  - Block tree: O(log n) steps, minimal shared memory traffic\n");
  printf("  - Avoid atomic contention in inner loops\n");

  // Cleanup
  free(h_input);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
