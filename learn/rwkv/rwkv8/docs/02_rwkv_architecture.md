# RWKV 架构详解

## 1. 整体架构

RWKV 采用 Transformer-like 的块结构，但内部替换为 WKV 注意力机制：

```
Input
  │
  ├─→ LayerNorm ──→ Time Mixing (WKV) ──→ Residual ─┐
  │                                                   ├─→ Output
  ├─→ LayerNorm ──→ Channel Mixing (FFN) ──→ Residual ┘
  │
  ↓
Next Layer (堆叠 N 层)
```

## 2. Time Mixing 层详解

### 2.1 时间混合机制

RWKV 引入时间衰减概念，当前位置只接收之前位置的部分信息：

```python
def time_mixing(x, last_x, decay, bonus):
    """
    x: 当前输入 [batch, dim]
    last_x: 上一时刻输入 (用于时间混合)
    decay: 衰减率，控制历史信息保留程度
    bonus: 当前位置额外权重
    """
    # 时间混合: 混合当前和前一时间步
    mixed = x * time_mix_w + last_x * (1 - time_mix_w)
    
    # 计算 R, K, V
    r = sigmoid(mixed @ W_r)  # receptance: 接收门
    k = mixed @ W_k           # key
    v = mixed @ W_v           # value
    
    # WKV 计算 (核心)
    wkv = compute_wkv(r, k, v, decay, bonus)
    
    return wkv
```

### 2.2 WKV 计算的四种变体

#### 版本 1: 基础并行版本 (训练)

```python
def wkv_parallel(k, v, w, u):
    """
    并行计算，适合训练
    k, v: [batch, len, dim]
    w: [dim] - 衰减率 (负数)
    u: [dim] - bonus 项
    """
    B, T, C = k.shape
    
    # 创建衰减矩阵
    # D[i,j] = exp(w)^(i-j) for i >= j, else 0
    w_vec = torch.exp(w)  # 衰减因子
    
    # 累积和用于计算衰减权重
    w_cumsum = torch.cumsum(w.expand(T, C), dim=0)  # [T, C]
    
    # D[i,j] = exp(w_cumsum[i] - w_cumsum[j]) for j <= i
    # 使用广播创建下三角矩阵
    diff = w_cumsum.unsqueeze(1) - w_cumsum.unsqueeze(0)  # [T, T, C]
    D = torch.exp(diff)  # 衰减矩阵
    D = torch.tril(D)  # 只保留下三角（causal）
    
    # 计算加权分子和分母
    exp_k = torch.exp(k)  # [B, T, C]
    
    # 分子: sum_j D[i,j] * exp(k_j) * v_j
    numerator = torch.einsum('btd,ijt,btd->bid', exp_k * v, D, torch.ones_like(D))
    # 简化: 对每个位置 i，计算 sum_{j<=i} exp(w*(i-j) + k_j) * v_j
    
    # 更高效的方式: 使用 cumsum
    exp_w = torch.exp(w)  # 衰减因子
    
    # 反向累积，实现类似 attention 的效果
    num = torch.zeros_like(v)
    den = torch.zeros_like(k)
    
    # 从后向前累积（或者从前向后，取决于定义）
    for t in range(T):
        if t == 0:
            num[:, t] = exp_k[:, t] * v[:, t]
            den[:, t] = exp_k[:, t]
        else:
            num[:, t] = exp_k[:, t] * v[:, t] + num[:, t-1] * exp_w
            den[:, t] = exp_k[:, t] + den[:, t-1] * exp_w
    
    # 应用 bonus 项（给当前位置额外权重）
    num_bonus = num + torch.exp(k + u.unsqueeze(0).unsqueeze(0)) * v
    den_bonus = den + torch.exp(k + u.unsqueeze(0).unsqueeze(0))
    
    wkv = num_bonus / (den_bonus + 1e-6)
    
    return wkv
```

#### 版本 2: 高效并行版本 (实际使用)

```python
def wkv_parallel_fast(r, k, v, w, u):
    """
    优化的并行 WKV 实现
    
    r: [B, T, C] - receptance (gate)
    k: [B, T, C] - key
    v: [B, T, C] - value
    w: [C] - time decay (负数)
    u: [C] - bonus (当前位置额外权重)
    """
    B, T, C = k.shape
    
    # 确保 w 是负数
    w = -torch.abs(w)
    
    # 计算 exp(k) 用于后续
    exp_k = torch.exp(k)  # [B, T, C]
    
    # 创建累积衰减矩阵
    # cumsum_w[t] = sum_{i=0}^{t-1} w = w * t
    # 实际上我们需要: w_cumsum[i] - w_cumsum[j] = w * (i - j)
    t_range = torch.arange(T, device=k.device).float()  # [T]
    
    # 计算 D[i,j] = exp(w * (i - j)) for j <= i
    # D[i,j] = exp(w)^|i-j| for causal attention
    w_exp = torch.exp(w)  # [C]
    
    # 使用累积和实现高效计算
    # numerator[t] = sum_{i=0}^t exp(w*(t-i) + k_i) * v_i
    #             = sum_{i=0}^t (exp(w)^(t-i) * exp(k_i)) * v_i
    
    # 高效实现: 使用卷积或累积
    num = torch.zeros(B, T, C, device=k.device)
    den = torch.zeros(B, T, C, device=k.device)
    
    # 从前往后累积
    for t in range(T):
        if t == 0:
            num[:, t] = exp_k[:, t] * v[:, t]
            den[:, t] = exp_k[:, t]
        else:
            # num_t = exp(k_t) * v_t + exp(w) * num_{t-1}
            # den_t = exp(k_t) + exp(w) * den_{t-1}
            num[:, t] = exp_k[:, t] * v[:, t] + w_exp * num[:, t-1]
            den[:, t] = exp_k[:, t] + w_exp * den[:, t-1]
    
    # 应用 bonus (给当前位置额外权重)
    exp_k_bonus = torch.exp(k + u.unsqueeze(0).unsqueeze(0))  # [B, T, C]
    num_bonus = num + exp_k_bonus * v
    den_bonus = den + exp_k_bonus
    
    # 应用 receptance gate
    wkv = r * num_bonus / (den_bonus + 1e-6)
    
    return wkv
```

#### 版本 3: RNN 形式 (推理时使用)

```python
class WKVCell(nn.Module):
    """
    RNN 形式的 WKV，用于自回归推理
    只维护状态，不保存历史
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 可学习参数
        self.time_decay = nn.Parameter(torch.randn(dim))  # w
        self.time_bonus = nn.Parameter(torch.randn(dim))  # u
        
        # 投影层
        self.receptance = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x, state=None):
        """
        x: [B, C] - 当前输入
        state: tuple (num, den) 或 None
        returns: (output, new_state)
        """
        B, C = x.shape
        
        # 初始化状态
        if state is None:
            num = torch.zeros(B, C, device=x.device)
            den = torch.zeros(B, C, device=x.device)
        else:
            num, den = state
        
        # 计算 R, K, V
        r = torch.sigmoid(self.receptance(x))
        k = self.key(x)
        v = self.value(x)
        
        # 确保 w 是负数
        w = -torch.abs(self.time_decay)
        u = self.time_bonus
        
        # WKV 计算（单步）
        exp_k = torch.exp(k)
        exp_w = torch.exp(w)
        
        # 新状态
        num_new = exp_k * v + exp_w * num
        den_new = exp_k + exp_w * den
        
        # 应用 bonus
        exp_k_u = torch.exp(k + u)
        num_bonus = num_new + exp_k_u * v
        den_bonus = den_new + exp_k_u
        
        # 最终输出
        out = r * num_bonus / (den_bonus + 1e-6)
        
        return out, (num_new, den_new)
```

## 4. RWKV-8 (Eagle) 的改进

RWKV-8 引入了 Multi-Head Latent Attention (MLA) 的概念：

```python
class RWKV8MLA(nn.Module):
    """
    Multi-Head Latent Attention for RWKV-8
    压缩 Key/Value 到 latent space，大幅减少内存
    """
    def __init__(self, dim, num_heads, latent_dim):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim  # 通常远小于 num_heads * head_dim
        
        # 低秩压缩矩阵
        self.k_down = nn.Linear(dim, latent_dim)   # 压缩 K
        self.k_up = nn.Linear(latent_dim, dim)     # 恢复 K
        self.v_down = nn.Linear(dim, latent_dim)   # 压缩 V
        self.v_up = nn.Linear(latent_dim, dim)     # 恢复 V
        
        # 其他参数
        self.r_proj = nn.Linear(dim, dim)
        self.time_decay = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.time_bonus = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        
        # 计算 receptance
        r = torch.sigmoid(self.r_proj(x))
        
        # 压缩并恢复 K, V
        k_latent = self.k_down(x)  # [B, T, latent_dim]
        k = self.k_up(k_latent)     # [B, T, D]
        
        v_latent = self.v_down(x)  # [B, T, latent_dim]
        v = self.v_up(v_latent)     # [B, T, D]
        
        # 分头计算 WKV
        r = r.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        
        # [B, H, T, D//H]
        r = r.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 使用高效 WKV 计算 (并行)
        wkv = self.wkv_parallel(r, k, v)
        
        # 合并头
        wkv = wkv.transpose(1, 2).contiguous().view(B, T, D)
        
        return wkv
    
    def wkv_parallel(self, r, k, v):
        """
        优化的并行 WKV，支持多头
        r, k, v: [B, H, T, D]
        """
        B, H, T, D = k.shape
        
        # 获取每头的参数
        w = -torch.abs(self.time_decay)  # [H, D]
        u = self.time_bonus               # [H, D]
        
        # 扩展维度用于广播
        w = w.view(1, H, 1, D)  # [1, H, 1, D]
        u = u.view(1, H, 1, D)
        
        # 计算
        exp_k = torch.exp(k)  # [B, H, T, D]
        exp_w = torch.exp(w)  # [1, H, 1, D]
        
        # 累积计算 numerator 和 denominator
        num = torch.zeros(B, H, D, device=k.device)
        den = torch.zeros(B, H, D, device=k.device)
        
        outputs = []
        
        for t in range(T):
            num = exp_k[:, :, t, :] * v[:, :, t, :] + num * exp_w
            den = exp_k[:, :, t, :] + den * exp_w
            
            # Bonus
            exp_k_u = torch.exp(k[:, :, t, :] + u)
            num_bonus = num + exp_k_u * v[:, :, t, :]
            den_bonus = den + exp_k_u
            
            out = r[:, :, t, :] * num_bonus / (den_bonus + 1e-6)
            outputs.append(out)
        
        return torch.stack(outputs, dim=2)
```

## 5. 性能优化总结

### 5.1 各架构的优化策略对比

| 优化技术 | RWKV | Mamba | DeltaNet |
|---------|------|-------|----------|
| **并行训练** | 前缀和 cumsum | 硬件感知扫描 | 类似 RWKV |
| **递推推理** | O(1) 状态 | O(1) 状态 | O(1) 状态 |
| **内存优化** | 无 KV cache | 固定状态大小 | 固定状态大小 |
| **长序列** | 线性复杂度 | 线性 + 选择性 | 线性 + 增量 |
| **硬件优化** | CUDA 分块 | FlashAttention 风格 | 类似 RWKV |

### 5.2 关键优化技术

```
1. 分块计算 (Tiling)
   - 将序列分成小块在 SRAM 中处理
   - 减少对 HBM 的访问

2. 重新计算 (Recomputation)
   - 反向传播时重新计算前向的中间结果
   - 用计算换内存

3. 内核融合 (Kernel Fusion)
   - 合并多个操作到一个 CUDA kernel
   - 减少 kernel launch 开销和内存往返

4. Warp-level 优化
   - 使用 warp shuffle 进行快速数据交换
   - 共享内存用于 warp 间通信
```

## 6. 下一步

- 查看 `reference/` 中的完整 PyTorch 实现
- 学习 `cuda/` 中的高性能 CUDA 内核
- 运行 `benchmarks/` 中的性能对比测试
- 阅读 `experiments/` 中的消融实验
