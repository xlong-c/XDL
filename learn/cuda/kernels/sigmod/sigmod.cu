#include <cuda_runtime.h>
#include <stdio.h>

/*
sigmod:
    f(x) = 1/(1+e^(-x))
*/

#define WARP_SIZE 32
#define E 2.718281828459045

__global__ void sigmod_simple(float *input, float *output, int mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mask) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

int main() {
    const int N = 1024;
    const int mask = N;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Host memory allocation
    float *h_input = new float[N];
    float *h_output = new float[N];

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i / N - 0.5f;  // Range: -0.5 to 0.5
    }

    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    sigmod_simple<<<gridSize, blockSize>>>(d_input, d_output, mask);

    // Copy output to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first 10 results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("sigmod(%.2f) = %.6f\n", h_input[i], h_output[i]);
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}