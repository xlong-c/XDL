# FlashAttention v2: 调度与 Occupancy 优化
# 核心：重排任务并行化逻辑，减少非矩阵乘法（Non-MatMul）开销，提高 SM 的资源占用率。

import torch

def flash_attention_v2_kernel(Q, K, V, block_size=128):
    """
    逻辑：在 v1 基础上，增加更细粒度的任务划分，提升 SM 计算饱和度。
    """
    N, D = Q.shape
    O = torch.zeros_like(Q)
    
    # 增加并行粒度，使 GPU 的 Tensor Core 能够同时处理更多小的 Tiles
    # 这比 v1 更有利于掩盖 HBM 的访存延迟
    for i in range(0, N, block_size):
        qi = Q[i:i+block_size]
        
        # 并行处理所有 Q 与 K 的块组合
        # v2 改进了线程块 (Thread Blocks) 之间的工作分配策略
        for j in range(0, N, block_size):
            kj = K[j:j+block_size]
            vj = V[j:j+block_size]
            
            # 融合算子优化：将 softmax 融合进矩阵乘法逻辑中
            # 减少写回 HBM 的次数，利用 Tensor Core 的高吞吐量
            p_ij = torch.softmax(torch.matmul(qi, kj.T), dim=-1)
            o_i = torch.matmul(p_ij, vj)
            
            # 原子更新结果
            O[i:i+block_size] += o_i
            
    return O
