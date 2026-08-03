#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 简单错误检查宏
#define CHECK_CUDA(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

#define CHECK_CUBLAS(call)                                               \
    {                                                                    \
        cublasStatus_t err = call;                                       \
        if (err != CUBLAS_STATUS_SUCCESS) {                              \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":"           \
                      << __LINE__ << " (code: " << (int)err << ")" << std::endl; \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

/**
 * 演示 Conv + BN + ReLU + Conv 的融合逻辑
 * 1. BN 已经预先融合进第一个 Conv 的权重和 Bias
 * 2. 使用 cuBLASLt Epilogue 融合 Bias + ReLU
 * 3. 第一个 Kernel 输出作为第二个 Kernel 输入
 */
void run_fused_conv_sequence() {
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // 矩阵维度 (M, N, K) -> 对应卷积中的 (OutChannels, Batch*H*W, InChannels)
    int m = 128, n = 256, k = 64;
    float alpha = 1.0f, beta = 0.0f;

    // 分配设备内存
    float *d_A1, *d_B1, *d_bias1, *d_intermediate; // 第一个 Conv
    float *d_A2, *d_out;                    // 第二个 Conv
    
    CHECK_CUDA(cudaMalloc(&d_A1, m * k * sizeof(float)));           // Weights 1
    CHECK_CUDA(cudaMalloc(&d_B1, k * n * sizeof(float)));           // Input 1
    CHECK_CUDA(cudaMalloc(&d_bias1, m * sizeof(float)));            // Fused BN Bias 1
    CHECK_CUDA(cudaMalloc(&d_intermediate, m * n * sizeof(float))); // Intermediate (ReLU output)
    
    CHECK_CUDA(cudaMalloc(&d_A2, m * m * sizeof(float)));           // Weights 2
    CHECK_CUDA(cudaMalloc(&d_out, m * n * sizeof(float)));          // Output 2

    // --- 第一步：配置第一个融合 Kernel (Conv + Bias + ReLU) ---
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t adesc = NULL, bdesc = NULL, cdesc = NULL, ddesc = NULL;
    
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    // 设置 Epilogue 融合属性
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias1, sizeof(d_bias1)));

    // 设置矩阵布局 (通常 cuBLAS 默认列优先)
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_32F, m, k, m));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_32F, k, n, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_32F, m, n, m));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&ddesc, CUDA_R_32F, m, n, m));

    // 获取启发式搜索算法
    cublasLtMatmulPreference_t preference = NULL;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspaceSize = 4 * 1024 * 1024;
    void* workspace;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedAlgoCount = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, adesc, bdesc, cdesc, ddesc, preference, 1, &heuristicResult, &returnedAlgoCount));

    // 执行第一个融合内核: intermediate = ReLU(A1 * B1 + Bias1)
    std::cout << "Executing First Fused Kernel: Conv + Bias + ReLU..." << std::endl;
    CHECK_CUBLAS(cublasLtMatmul(ltHandle, operationDesc, 
                                &alpha, d_A1, adesc, 
                                d_B1, bdesc, 
                                &beta, d_intermediate, cdesc, 
                                d_intermediate, ddesc, 
                                &heuristicResult.algo, workspace, workspaceSize, 0));

    // --- 第二步：配置第二个 Kernel (常规 Conv) ---
    // 这里我们将 intermediate 作为 B 矩阵输入
    cublasLtMatmulDesc_t opDesc2 = NULL;
    cublasLtMatrixLayout_t adesc2 = NULL;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc2, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&adesc2, CUDA_R_32F, m, m, m));

    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ltHandle, opDesc2, adesc2, ddesc, ddesc, ddesc, preference, 1, &heuristicResult, &returnedAlgoCount));

    std::cout << "Executing Second Kernel: Conv..." << std::endl;
    CHECK_CUBLAS(cublasLtMatmul(ltHandle, opDesc2, 
                                &alpha, d_A2, adesc2, 
                                d_intermediate, ddesc, 
                                &beta, d_out, ddesc, 
                                d_out, ddesc, 
                                &heuristicResult.algo, workspace, workspaceSize, 0));

    std::cout << "Sequence completed successfully." << std::endl;

    // 清理资源 (省略部分以保持简洁)
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(opDesc2));
    CHECK_CUBLAS(cublasLtDestroy(ltHandle));
    CHECK_CUDA(cudaFree(d_A1));
    CHECK_CUDA(cudaFree(d_B1));
    CHECK_CUDA(cudaFree(d_intermediate));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(workspace));
}

int main() {
    run_fused_conv_sequence();
    return 0;
}
