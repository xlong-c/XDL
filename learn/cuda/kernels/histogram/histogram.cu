#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

#ifdef PYTORCH_EXTENSION
#include <torch/extension.h>
#include <torch/types.h>
#endif

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

//Histogram

__global__ void histogram_i32_kernel(const int *input, int *hist, int n, int nbin) { 
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
          atomicAdd(hist + input[i], 1);
      }
}

// int4
__global__ void histogram_i4_kernel(const int4 *input, int *hist, int n, int nbin) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
          atomicAdd(hist + input[i].x, 1);
          atomicAdd(hist + input[i].y, 1);
          atomicAdd(hist + input[i].z, 1);
          atomicAdd(hist + input[i].w, 1);
      }
}

// CUDA error checking
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// CPU reference implementation for verification
void histogram_cpu(const int* input, int* hist, int n, int nbin) {
    for (int i = 0; i < nbin; i++) hist[i] = 0;
    for (int i = 0; i < n; i++) {
        int val = input[i];
        if (val >= 0 && val < nbin) {
            hist[val]++;
        }
    }
}

// Benchmark function for histogram kernels
void benchmark_histogram(int n, int nbin) {
    printf("=== Histogram Benchmark: n=%d, nbin=%d ===\n", n, nbin);

    // Allocate host memory
    size_t input_size = n * sizeof(int);
    size_t hist_size = nbin * sizeof(int);

    int* h_input = (int*)malloc(input_size);
    int* h_hist_cpu = (int*)malloc(hist_size);
    int* h_hist_gpu_i32 = (int*)malloc(hist_size);
    int* h_hist_gpu_i4 = (int*)malloc(hist_size);

    // Initialize random input (0 to nbin-1)
    srand(12345);
    for (int i = 0; i < n; i++) {
        h_input[i] = rand() % nbin;
    }

    // Compute CPU reference
    histogram_cpu(h_input, h_hist_cpu, n, nbin);

    // Allocate device memory
    int* d_input;
    int* d_hist_i32;
    int* d_hist_i4;
    checkCudaError(cudaMalloc(&d_input, input_size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_hist_i32, hist_size), "cudaMalloc d_hist_i32");
    checkCudaError(cudaMalloc(&d_hist_i4, hist_size), "cudaMalloc d_hist_i4");

    // Copy input to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice),
                   "cudaMemcpy d_input");

    // Test i32 kernel
    {
        // Clear histogram
        checkCudaError(cudaMemset(d_hist_i32, 0, hist_size), "cudaMemset d_hist_i32");

        // Launch kernel
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        cudaEvent_t start, stop;
        checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
        checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

        // Warmup
        histogram_i32_kernel<<<grid_size, block_size>>>(d_input, d_hist_i32, n, nbin);
        cudaDeviceSynchronize();

        // Timing
        checkCudaError(cudaEventRecord(start), "cudaEventRecord start");
        int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            // Clear histogram before each iteration
            checkCudaError(cudaMemset(d_hist_i32, 0, hist_size), "cudaMemset d_hist_i32");
            histogram_i32_kernel<<<grid_size, block_size>>>(d_input, d_hist_i32, n, nbin);
        }
        checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
        checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float milliseconds = 0;
        checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
        float avg_ms = milliseconds / iterations;

        printf("i32 Kernel: %.3f ms (avg over %d iterations)\n", avg_ms, iterations);

        // Copy result back
        checkCudaError(cudaMemcpy(h_hist_gpu_i32, d_hist_i32, hist_size, cudaMemcpyDeviceToHost),
                      "cudaMemcpy h_hist_gpu_i32");

        // Verify
        bool correct = true;
        for (int i = 0; i < nbin; i++) {
            if (h_hist_gpu_i32[i] != h_hist_cpu[i]) {
                printf("  Verification FAILED at bin %d: GPU=%d, CPU=%d\n",
                       i, h_hist_gpu_i32[i], h_hist_cpu[i]);
                correct = false;
                break;
            }
        }
        if (correct) printf("  Verification: PASS\n");

        checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
        checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    }

    // Test i4 kernel
    {
        // Clear histogram
        checkCudaError(cudaMemset(d_hist_i4, 0, hist_size), "cudaMemset d_hist_i4");

        // For i4 kernel, we need to reinterpret input as int4*
        int n_i4 = n / 4;  // Number of int4 elements
        const int4* d_input_i4 = reinterpret_cast<const int4*>(d_input);

        // Launch kernel
        int block_size = 256;
        int grid_size = (n_i4 + block_size - 1) / block_size;

        cudaEvent_t start, stop;
        checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
        checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

        // Warmup
        histogram_i4_kernel<<<grid_size, block_size>>>(d_input_i4, d_hist_i4, n_i4, nbin);
        cudaDeviceSynchronize();

        // Timing
        checkCudaError(cudaEventRecord(start), "cudaEventRecord start");
        int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            // Clear histogram before each iteration
            checkCudaError(cudaMemset(d_hist_i4, 0, hist_size), "cudaMemset d_hist_i4");
            histogram_i4_kernel<<<grid_size, block_size>>>(d_input_i4, d_hist_i4, n_i4, nbin);
        }
        checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
        checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float milliseconds = 0;
        checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
        float avg_ms = milliseconds / iterations;

        printf("i4 Kernel: %.3f ms (avg over %d iterations)\n", avg_ms, iterations);

        // Copy result back
        checkCudaError(cudaMemcpy(h_hist_gpu_i4, d_hist_i4, hist_size, cudaMemcpyDeviceToHost),
                      "cudaMemcpy h_hist_gpu_i4");

        // Verify
        bool correct = true;
        for (int i = 0; i < nbin; i++) {
            if (h_hist_gpu_i4[i] != h_hist_cpu[i]) {
                printf("  Verification FAILED at bin %d: GPU=%d, CPU=%d\n",
                       i, h_hist_gpu_i4[i], h_hist_cpu[i]);
                correct = false;
                break;
            }
        }
        if (correct) printf("  Verification: PASS\n");

        checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
        checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_hist_i32);
    cudaFree(d_hist_i4);
    free(h_input);
    free(h_hist_cpu);
    free(h_hist_gpu_i32);
    free(h_hist_gpu_i4);

    printf("\n");
}

#ifndef PYTORCH_EXTENSION

int main() {
    // Test different sizes
    benchmark_histogram(1024 * 1024, 256);      // 1M elements, 256 bins
    benchmark_histogram(1024 * 1024 * 4, 512);  // 4M elements, 512 bins
    benchmark_histogram(1024 * 1024 * 16, 1024);// 16M elements, 1024 bins

    return 0;
}

#else

// PyTorch CUDA extension for histogram

void histogram_i32_cuda_forward(torch::Tensor input, torch::Tensor hist, int nbin) {
    int n = input.size(0);
    int* d_input = input.data_ptr<int>();
    int* d_hist = hist.data_ptr<int>();

    // Clear histogram
    cudaError_t err = cudaMemset(d_hist, 0, nbin * sizeof(int));
    TORCH_CHECK(err == cudaSuccess, "cudaMemset failed: ", cudaGetErrorString(err));

    // Launch kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    histogram_i32_kernel<<<grid_size, block_size>>>(d_input, d_hist, n, nbin);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "histogram_i32_kernel launch failed: ", cudaGetErrorString(err));
}

void histogram_i4_cuda_forward(torch::Tensor input, torch::Tensor hist, int nbin) {
    int n = input.size(0);
    // Input must be contiguous and have size divisible by 4
    TORCH_CHECK(n % 4 == 0, "Input size must be divisible by 4 for i4 kernel");

    const int4* d_input = reinterpret_cast<const int4*>(input.data_ptr<int>());
    int* d_hist = hist.data_ptr<int>();

    // Clear histogram
    cudaError_t err = cudaMemset(d_hist, 0, nbin * sizeof(int));
    TORCH_CHECK(err == cudaSuccess, "cudaMemset failed: ", cudaGetErrorString(err));

    // Launch kernel
    int block_size = 256;
    int grid_size = (n / 4 + block_size - 1) / block_size;
    histogram_i4_kernel<<<grid_size, block_size>>>(d_input, d_hist, n / 4, nbin);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "histogram_i4_kernel launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("histogram_i32_forward", &histogram_i32_cuda_forward, "Histogram forward (i32 CUDA)");
    m.def("histogram_i4_forward", &histogram_i4_cuda_forward, "Histogram forward (i4 CUDA)");
}

#endif