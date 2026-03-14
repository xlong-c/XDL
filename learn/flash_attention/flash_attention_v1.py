# FlashAttention v1: 分块 (Tiling) 与 算子融合
# 核心突破：通过将计算切分为 SRAM 大小的块，减少 Global Memory (HBM) 的 IO 开销。

import torch

def flash_attention_v1_kernel(Q, K, V, block_size=128):
    """
    Q, K, V: [N, D] - 查询、键、值矩阵
    逻辑：将计算切分为小块，使每个块能完整放入 SRAM。
    """
    N, D = Q.shape
    # O 为输出矩阵，存储在 HBM
    O = torch.zeros_like(Q)
    
    # 循环分块 (Tiling)
    # 减少了 HBM 与 SRAM 之间的高频读写，这是 v1 的核心贡献
    for i in range(0, N, block_size):
        qi = Q[i:i+block_size]
        
        # 将块从 HBM 加载到 SRAM
        ki = K[i:i+block_size]
        vi = V[i:i+block_size]
        
        # 计算该块的 Attention 分数
        # S_i = softmax(Q_i * K_i^T)
        S_i = torch.softmax(torch.matmul(qi, ki.T), dim=-1)
        
        # 累加输出
        O[i:i+block_size] = torch.matmul(S_i, vi)
        
    return O
