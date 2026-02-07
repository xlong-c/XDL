# RWKV WKV 计算的数学推导

## 1. 传统 Softmax Attention 的问题

### 1.1 标准计算

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**问题：**
- $QK^T$ 的复杂度是 $O(N^2)$
- 需要存储 $N \times N$ 的注意力矩阵
- 内存和计算随序列长度平方增长

### 1.2 KV Cache 问题

推理时需要缓存所有历史 K 和 V：

```python
# Transformer KV Cache
kv_cache_size = 2 * num_layers * num_heads * seq_len * head_dim * batch_size * sizeof(float)

# 例如: 32层, 8头, 32768长度, 128维, batch=1, fp16
# 内存 = 2 * 32 * 8 * 32768 * 128 * 2 bytes ≈ 8.5 GB
```

## 2. RWKV 的核心思想

### 2.1 从 Softmax 到线性注意力

**传统 Softmax：**

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

**线性注意力变体：**

如果用 $e^x \approx 1 + x$ 或替换核函数：

$$
\text{sim}(q, k) = \phi(q)^T \phi(k) \quad \text{其中} \phi(x) = \text{elu}(x) + 1
$$

则：

$$
\text{Attention}(Q, K, V) = \frac{\sum_{j \leq i} \phi(q_i)^T \phi(k_j) \cdot v_j}{\sum_{j \leq i} \phi(q_i)^T \phi(k_j)}
$$

这可以改写为递推形式！

### 2.2 RWKV 的递推公式

RWKV 定义了四个关键向量：

- **R** (Receptance): 接收向量，控制当前 token 能"接收"多少历史信息
- **W** (Weight): 位置权重衰减，实现随距离指数衰减
- **K** (Key): 键向量，类似 Transformer
- **V** (Value): 值向量，类似 Transformer

**核心 WKV 计算：**

$$
a_t = \sum_{i=1}^{t} e^{-(t-i)w + k_i} \cdot v_i \\
b_t = \sum_{i=1}^{t} e^{-(t-i)w + k_i} \\
\text{wkv}_t = \frac{r_t \cdot a_t}{b_t}
$$

**递推形式（O(1) 推理）：**

$$
a_t = e^{-w + k_t} \cdot v_t + e^{-w} \cdot a_{t-1} \\
b_t = e^{-w + k_t} + e^{-w} \cdot b_{t-1}
$$

### 2.3 RWKV 的并行训练

虽然推理是 RNN 形式，但训练可以并行：

```python
def rwkv_parallel(W, K, V, B, T):
    """
    W: [B, T, C] - decay rates
    K: [B, T, C] - keys  
    V: [B, T, C] - values
    B: [B, T, C] - receptance (gating)
    
    使用前缀和 (cumsum) 实现并行计算
    """
    # 计算衰减矩阵 [B, T, T, C]
    w = -torch.exp(W)  # 确保衰减为负
    
    # 累积衰减: e^{w_1 + w_2 + ... + w_t}
    w_cumsum = torch.cumsum(w, dim=1)  # [B, T, C]
    
    # 创建衰减矩阵 D[i,j] = e^{w_{j+1} + ... + w_i} for j <= i
    D = torch.exp(w_cumsum.unsqueeze(1) - w_cumsum.unsqueeze(2))  # [B, T, T, C]
    D = torch.tril(D)  # 下三角矩阵 (causal)
    
    # 计算加权值
    V_weighted = D * K.unsqueeze(-1) * V.unsqueeze(2)  # [B, T, T, C]
    numerator = V_weighted.sum(dim=2)  # [B, T, C]
    
    denominator = (D * K.unsqueeze(-1)).sum(dim=2)  # [B, T, C]
    
    # 应用 receptance gating
    output = B * numerator / (denominator + 1e-6)
    
    return output
```

## 3. 数值稳定性处理

### 3.1 Softmax 数值稳定技巧

```python
def stable_softmax(x, dim=-1):
    """
    数值稳定的 softmax
    subtract max for numerical stability
    """
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max  # Now max(x) = 0
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
```

### 3.2 WKV 数值稳定版本

```python
def wkv_forward_stable(r, k, v, w, u):
    """
    数值稳定的 WKV 计算
    
    r: [B, T, C] - receptance
    k: [B, T, C] - key
    v: [B, T, C] - value
    w: [C] - decay rate (shared across batch/time)
    u: [C] - bonus term for current position
    """
    B, T, C = r.shape
    
    # 确保数值稳定性
    w = -torch.exp(w)  # w < 0 确保衰减
    
    # 初始化状态
    num = torch.zeros(B, C, device=r.device)  # numerator
    den = torch.zeros(B, C, device=r.device)  # denominator
    
    outputs = []
    
    for t in range(T):
        # 当前时间步
        rt = r[:, t, :]  # [B, C]
        kt = k[:, t, :]
        vt = v[:, t, :]
        
        # 数值稳定: 使用 log-space 或缩放
        # 计算当前位置的贡献
        e_k = torch.exp(kt)  # exp(k_t)
        
        # 更新状态 (应用衰减)
        num = num * torch.exp(w) + e_k * vt
        den = den * torch.exp(w) + e_k
        
        # 应用 bonus term 并计算输出
        # bonus 给当前位置额外权重
        num_bonus = num + torch.exp(kt + u) * vt
        den_bonus = den + torch.exp(kt + u)
        
        out = rt * num_bonus / (den_bonus + 1e-6)
        outputs.append(out)
    
    return torch.stack(outputs, dim=1)
```

## 4. CUDA 优化要点

### 4.1 内存访问模式

```cuda
// 坏的内存访问模式（非合并访问）
__global__ void bad_kernel(float* data, int N) {
    int tid = threadIdx.x;
    for (int i = 0; i < N; i += 32) {
        // 每个线程访问非连续位置，无法合并
        data[i + tid * N / 32] = ...;  // 步长跳跃
    }
}

// 好的内存访问模式（合并访问）
__global__ void good_kernel(float* data, int N) {
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        // 连续线程访问连续地址
        data[i] = ...;  // 完全合并
    }
}
```

### 4.2 Shared Memory 使用

```cuda
// 使用 Shared Memory 避免重复全局内存访问
__global__ void wkv_shared_kernel(float* r, float* k, float* v, 
                                   float* w, float* u, float* out,
                                   int B, int T, int C) {
    // 每个 block 处理一个 batch 的一个 channel 块
    __shared__ float s_r[256];
    __shared__ float s_k[256];
    __shared__ float s_v[256];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int b = bid / C;
    int c = bid % C;
    
    if (b >= B) return;
    
    // 加载数据到 shared memory
    for (int t = tid; t < T; t += blockDim.x) {
        int idx = (b * T + t) * C + c;
        s_r[tid] = r[idx];
        s_k[tid] = k[idx];
        s_v[tid] = v[idx];
    }
    __syncthreads();
    
    // WKV 计算 (使用 shared memory 数据)
    float num = 0.0f;
    float den = 0.0f;
    float decay = expf(w[c]);
    
    for (int t = 0; t < T; t++) {
        float rt = s_r[t];
        float kt = s_k[t];
        float vt = s_v[t];
        
        float ek = expf(kt);
        num = num * decay + ek * vt;
        den = den * decay + ek;
        
        float bonus = expf(kt + u[c]);
        float num_b = num + bonus * vt;
        float den_b = den + bonus;
        
        int out_idx = (b * T + t) * C + c;
        out[out_idx] = rt * num_b / (den_b + 1e-6f);
    }
}
```

这只是一个概览，详细实现请查看各个目录下的完整代码和文档。
