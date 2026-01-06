/**
 * Softmax 公式:
 * Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * 执行流程 (Online Softmax):
 * 1. 局部在线归约 (Local Online Reduction): 每个线程处理多个元素，维护局部最大值 (max) 和累加和 (sum)。
 *    更新公式为: sum = sum * exp(old_max - new_max) + exp(new_val - new_max)。
 * 2. 块级归约 (Block-level Reduction): 使用 Warp Shuffle 原语和共享内存 (Shared Memory) 在整个 Block 内聚合局部 max/sum，
 *    得到该行最终的 final_max 和 final_sum。
 * 3. 归一化 (Normalization): 每个线程重新读取输入数据，计算 exp(val - final_max) / final_sum，并将结果写入全局内存。
 *
 * 优化点:
 * 1. Online Softmax: 将传统的三趟扫描（求最大值、求和、归一化）减少为两趟扫描，显著降低显存带宽压力。
 * 2. 向量化 (Vectorization): 使用 float4 (128位负载/存储) 来最大化内存带宽利用率。
 * 3. Warp Shuffle: 使用 __shfl_down_sync 进行快速的线程束内规约，避免了共享内存的访问延迟和同步开销。
 * 4. 循环展开与指针运算: 最小化循环内部的额外开销。
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <float.h>
#include <chrono>

#ifdef PYTORCH_EXTENSION
#include <torch/extension.h>
#endif

#define WARP_SIZE 32

// Warp-level reduction for max and sum in Online Softmax
__device__ __forceinline__ void warpReduceOnline(float &max_val, float &sum_val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        float remote_max = __shfl_down_sync(0xffffffff, max_val, offset);
        float remote_sum = __shfl_down_sync(0xffffffff, sum_val, offset);
        if (remote_max > max_val)
        {
            sum_val = sum_val * expf(max_val - remote_max) + remote_sum;
            max_val = remote_max;
        }
        else
        {
            sum_val += remote_sum * expf(remote_max - max_val);
        }
    }
}

// Optimized Online Softmax Kernel with Vectorization
// Processes 4 elements at a time using float4
__global__ void softmax_kernel_optimized(const float *__restrict__ input, float *__restrict__ output, int N)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int BLOCK_SIZE = blockDim.x;

    const float *row_input = input + row * N;
    float *row_output = output + row * N;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // 1. Vectorized Online reduction
    const float4 *row_input_vec = reinterpret_cast<const float4 *>(row_input);
    int num_vec = N / 4;

    for (int i = tid; i < num_vec; i += BLOCK_SIZE)
    {
        float4 val4 = row_input_vec[i];
        float vals[4] = {val4.x, val4.y, val4.z, val4.w};
        for (int j = 0; j < 4; ++j)
        {
            float val = vals[j];
            if (val > local_max)
            {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            }
            else
            {
                local_sum += expf(val - local_max);
            }
        }
    }

    // Handle remainder
    for (int i = num_vec * 4 + tid; i < N; i += BLOCK_SIZE)
    {
        float val = row_input[i];
        if (val > local_max)
        {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        }
        else
        {
            local_sum += expf(val - local_max);
        }
    }

    // 2. Block-level reduction
    warpReduceOnline(local_max, local_sum);

    static __shared__ float shared_max[WARP_SIZE];
    static __shared__ float shared_sum[WARP_SIZE];

    if (tid % WARP_SIZE == 0)
    {
        shared_max[tid / WARP_SIZE] = local_max;
        shared_sum[tid / WARP_SIZE] = local_sum;
    }
    __syncthreads();

    if (tid < WARP_SIZE)
    {
        local_max = (tid < BLOCK_SIZE / WARP_SIZE) ? shared_max[tid] : -FLT_MAX;
        local_sum = (tid < BLOCK_SIZE / WARP_SIZE) ? shared_sum[tid] : 0.0f;
        warpReduceOnline(local_max, local_sum);
        if (tid == 0)
        {
            shared_max[0] = local_max;
            shared_sum[0] = local_sum;
        }
    }
    __syncthreads();

    float final_max = shared_max[0];
    float final_sum = shared_sum[0];
    float inv_sum = 1.0f / final_sum;

    // 3. Final pass: Normalize and Store (Vectorized)
    float4 *row_output_vec = reinterpret_cast<float4 *>(row_output);
    for (int i = tid; i < num_vec; i += BLOCK_SIZE)
    {
        float4 val4 = row_input_vec[i];
        val4.x = expf(val4.x - final_max) * inv_sum;
        val4.y = expf(val4.y - final_max) * inv_sum;
        val4.z = expf(val4.z - final_max) * inv_sum;
        val4.w = expf(val4.w - final_max) * inv_sum;
        row_output_vec[i] = val4;
    }
    for (int i = num_vec * 4 + tid; i < N; i += BLOCK_SIZE)
    {
        row_output[i] = expf(row_input[i] - final_max) * inv_sum;
    }
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void benchmark(int rows, int cols)
{
    int N = rows * cols;
    size_t size = N * sizeof(float);

    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, size), "malloc d_input");
    checkCudaError(cudaMalloc(&d_output, size), "malloc d_output");
    checkCudaError(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice), "memcpy to device");

    int threads = 256;
    int blocks = rows;

    // Warmup
    softmax_kernel_optimized<<<blocks, threads>>>(d_input, d_output, cols);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 1000;
    for (int i = 0; i < iterations; ++i)
    {
        softmax_kernel_optimized<<<blocks, threads>>>(d_input, d_output, cols);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double avg_ms = diff.count() / iterations * 1000;
    double bandwidth = (2.0 * size) / (avg_ms / 1000.0) / 1e9; // Read + Write

    std::cout << "Size: " << rows << "x" << cols
              << " | Avg Time: " << avg_ms << " ms"
              << " | Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Verify
    std::vector<float> h_output(N);
    checkCudaError(cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost), "memcpy to host");
    for (int i = 0; i < std::min(rows, 5); ++i)
    {
        float sum = 0;
        for (int j = 0; j < cols; ++j)
            sum += h_output[i * cols + j];
        if (std::abs(sum - 1.0f) > 1e-4)
        {
            std::cout << "Verification FAILED at row " << i << " sum: " << sum << std::endl;
            return;
        }
    }
    std::cout << "Verification: PASS" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
}

#ifndef PYTORCH_EXTENSION

int main()
{

    benchmark(1024, 1024);

    benchmark(4096, 4096);

    benchmark(1024, 16384);

    return 0;
}

#else

void softmax_cuda_forward(torch::Tensor input, torch::Tensor output)
{

    const int rows = input.size(0);
    const int cols = input.size(1);
    int threads = 256;
    int blocks = rows;
    softmax_kernel_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        cols);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

    m.def("forward", &softmax_cuda_forward, "Softmax forward (CUDA)");
}

#endif
