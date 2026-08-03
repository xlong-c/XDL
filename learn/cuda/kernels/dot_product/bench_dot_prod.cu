#include "dot_product.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template<typename T>
void initialize_data(T* data, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            data[i] = dis(gen);
        } else if constexpr (std::is_same_v<T, half>) {
            data[i] = __float2half(dis(gen));
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            data[i] = __float2bfloat16(dis(gen));
        }
    }
}

float dot_product_cpu_f32(const float* a, const float* b, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += (double)a[i] * (double)b[i];
    }
    return (float)sum;
}

float dot_product_cpu_f16(const half* a, const half* b, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += (double)__half2float(a[i]) * (double)__half2float(b[i]);
    }
    return (float)sum;
}

float dot_product_cpu_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += (double)__bfloat162float(a[i]) * (double)__bfloat162float(b[i]);
    }
    return (float)sum;
}

void benchmark_f32(int N) {
    std::cout << "Benchmarking F32 Dot Product (N = " << N << ")" << std::endl;
    std::vector<float> h_a(N), h_b(N);
    initialize_data(h_a.data(), N);
    initialize_data(h_b.data(), N);

    float expected = dot_product_cpu_f32(h_a.data(), h_b.data(), N);

    float *d_a, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto run_kernel = [&](const char* name, auto kernel, int threads, int elements_per_block) {
        float h_y = 0.0f;
        int blocks = (N + elements_per_block - 1) / elements_per_block;
        
        // Correctness
        CUDA_CHECK(cudaMemcpy(d_y, &h_y, sizeof(float), cudaMemcpyHostToDevice));
        kernel<<<blocks, threads>>>(d_a, d_b, d_y, N);
        CUDA_CHECK(cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        
        float diff = std::abs(h_y - expected);
        float rel_err = diff / (std::abs(expected) + 1e-6f);
        bool passed = rel_err < 1e-4;

        // Timing
        const int iterations = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            kernel<<<blocks, threads>>>(d_a, d_b, d_y, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float avg_ms = milliseconds / iterations;

        double gflops = (2.0 * N) / (avg_ms * 1e-3) / 1e9;
        double gbs = (2.0 * N * sizeof(float)) / (avg_ms * 1e-3) / 1e9;

        std::cout << std::left << std::setw(30) << name 
                  << " | Error: " << std::scientific << std::setprecision(4) << rel_err 
                  << " | Pass: " << (passed ? "YES" : "NO")
                  << " | Time: " << std::fixed << std::setprecision(4) << avg_ms << " ms"
                  << " | GFLOPS: " << std::setprecision(2) << gflops
                  << " | GB/s: " << std::setprecision(2) << gbs << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    run_kernel("dot_prod_f32_f32_kernel", dot_prod_f32_f32_kernel, 256, 256);
    run_kernel("dot_prod_f32_f32_kernel_v2", dot_prod_f32_f32_kernel_v2, 256, 256 * 16);
    run_kernel("dot_prod_f32x4_f32_kernel", dot_prod_f32x4_f32_kernel, 256, 256 * 16);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_y);
}

void benchmark_f16(int N) {
    std::cout << "\nBenchmarking F16 Dot Product (N = " << N << ")" << std::endl;
    std::vector<half> h_a(N), h_b(N);
    initialize_data(h_a.data(), N);
    initialize_data(h_b.data(), N);

    float expected = dot_product_cpu_f16(h_a.data(), h_b.data(), N);

    half *d_a, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice));

    auto run_kernel = [&](const char* name, auto kernel, int threads, int elements_per_block) {
        half h_y = __float2half(0.0f);
        int blocks = (N + elements_per_block - 1) / elements_per_block;
        
        CUDA_CHECK(cudaMemcpy(d_y, &h_y, sizeof(half), cudaMemcpyHostToDevice));
        kernel<<<blocks, threads>>>(d_a, d_b, d_y, N);
        CUDA_CHECK(cudaMemcpy(&h_y, d_y, sizeof(half), cudaMemcpyDeviceToHost));
        
        float actual = __half2float(h_y);
        float diff = std::abs(actual - expected);
        float rel_err = diff / (std::abs(expected) + 1e-6f);
        bool passed = rel_err < 2e-2;

        const int iterations = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            kernel<<<blocks, threads>>>(d_a, d_b, d_y, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float avg_ms = milliseconds / iterations;

        double gflops = (2.0 * N) / (avg_ms * 1e-3) / 1e9;
        double gbs = (2.0 * N * sizeof(half)) / (avg_ms * 1e-3) / 1e9;

        std::cout << std::left << std::setw(30) << name 
                  << " | Error: " << std::scientific << std::setprecision(4) << rel_err 
                  << " | Pass: " << (passed ? "YES" : "NO")
                  << " | Time: " << std::fixed << std::setprecision(4) << avg_ms << " ms"
                  << " | GFLOPS: " << std::setprecision(2) << gflops
                  << " | GB/s: " << std::setprecision(2) << gbs << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    run_kernel("dot_prod_f16_f32_kernel", dot_prod_f16_f32_kernel, 256, 256 * 16 * 8);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_y);
}

void benchmark_bf16(int N) {
    std::cout << "\nBenchmarking BF16 Dot Product (N = " << N << ")" << std::endl;
    std::vector<__nv_bfloat16> h_a(N), h_b(N);
    initialize_data(h_a.data(), N);
    initialize_data(h_b.data(), N);

    float expected = dot_product_cpu_bf16(h_a.data(), h_b.data(), N);

    __nv_bfloat16 *d_a, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    auto run_kernel = [&](const char* name, auto kernel, int threads, int elements_per_block) {
        __nv_bfloat16 h_y = __float2bfloat16(0.0f);
        int blocks = (N + elements_per_block - 1) / elements_per_block;
        
        CUDA_CHECK(cudaMemcpy(d_y, &h_y, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        kernel<<<blocks, threads>>>(d_a, d_b, d_y, N);
        CUDA_CHECK(cudaMemcpy(&h_y, d_y, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        
        float actual = __bfloat162float(h_y);
        float diff = std::abs(actual - expected);
        float rel_err = diff / (std::abs(expected) + 1e-6f);
        bool passed = rel_err < 3e-1;

        const int iterations = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            kernel<<<blocks, threads>>>(d_a, d_b, d_y, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float avg_ms = milliseconds / iterations;

        double gflops = (2.0 * N) / (avg_ms * 1e-3) / 1e9;
        double gbs = (2.0 * N * sizeof(__nv_bfloat16)) / (avg_ms * 1e-3) / 1e9;

        std::cout << std::left << std::setw(30) << name 
                  << " | Error: " << std::scientific << std::setprecision(4) << rel_err 
                  << " | Pass: " << (passed ? "YES" : "NO")
                  << " | Time: " << std::fixed << std::setprecision(4) << avg_ms << " ms"
                  << " | GFLOPS: " << std::setprecision(2) << gflops
                  << " | GB/s: " << std::setprecision(2) << gbs << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    run_kernel("dot_prod_bf16x8_bf16_kernel", dot_prod_bf16x8_bf16_kernel, 256, 256 * 16 * 8);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_y);
}

int main() {
    int N = 1 << 24; 
    benchmark_f32(N);
    benchmark_f16(N);
    benchmark_bf16(N);
    return 0;
}