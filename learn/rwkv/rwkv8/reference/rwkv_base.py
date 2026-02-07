"""
RWKV 基础实现 - 纯 PyTorch，用于理解和验证算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RWKV_TimeMix(nn.Module):
    """
    RWKV Time Mixing 层 (WKV 机制)
    """
    def __init__(self, dim, layer_id=0):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        
        # 时间混合参数 - 用于混合当前和前一时间步
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_g = nn.Parameter(torch.ones(1, 1, dim))
        
        # WKV 参数
        self.time_decay = nn.Parameter(torch.ones(dim))  # w
        self.time_bonus = nn.Parameter(torch.ones(dim))   # u
        
        # 投影层
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        
        # Group Normalization
        self.ln = nn.GroupNorm(1, dim)
        
    def forward(self, x, state=None):
        """
        x: [batch, seq_len, dim]
        state: (prev_x, num, den) 用于推理时的状态传递
        """
        B, T, C = x.shape
        
        # 时间混合: 混合当前和前一时间步
        if state is not None:
            # 推理模式: 使用传入的状态
            prev_x = state[0]
            xx = torch.cat([prev_x.unsqueeze(1), x[:, :-1, :]], dim=1)
        else:
            # 训练模式: 通过 shift 实现
            xx = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        
        # 混合后的输入
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)
        
        # 计算 R, K, V, G
        r = torch.sigmoid(self.receptance(xr))  # receptance gate
        k = self.key(xk)
        v = self.value(xv)
        g = torch.sigmoid(self.gate(xg))
        
        # WKV 计算
        if state is not None:
            # 推理模式: 使用递推公式，维护状态
            wkv, new_state = self.wkv_forward_recursive(r, k, v, state[1:])
        else:
            # 训练模式: 并行计算
            wkv = self.wkv_forward_parallel(r, k, v)
            new_state = None
        
        # GroupNorm + gating
        wkv = self.ln(wkv.transpose(1, 2)).transpose(1, 2)
        wkv = wkv * g
        
        # 输出投影
        out = self.output(wkv)
        
        return out, new_state
    
    def wkv_forward_parallel(self, r, k, v):
        """
        并行 WKV 计算 (训练时使用)
        r, k, v: [batch, seq_len, dim]
        """
        B, T, C = k.shape
        
        # 确保 decay 是负数
        w = -torch.abs(self.time_decay)  # [C]
        u = self.time_bonus               # [C]
        
        # 计算 exp(k)
        exp_k = torch.exp(k)  # [B, T, C]
        exp_w = torch.exp(w)  # [C]
        
        # 累积计算 (使用 cumsum 实现并行)
        # 我们需要计算: num[t] = sum_{i=0}^t exp(w*(t-i) + k_i) * v_i
        
        # 方法: 使用 log-space 累积
        # 或者直接使用递推，但用向量化操作
        
        # 使用 torch.jit 优化的累积
        num = torch.zeros(B, T, C, device=k.device)
        den = torch.zeros(B, T, C, device=k.device)
        
        # 第一个位置
        num[:, 0] = exp_k[:, 0] * v[:, 0]
        den[:, 0] = exp_k[:, 0]
        
        # 累积 (可以向量化优化)
        for t in range(1, T):
            num[:, t] = exp_k[:, t] * v[:, t] + num[:, t-1] * exp_w
            den[:, t] = exp_k[:, t] + den[:, t-1] * exp_w
        
        # 应用 bonus
        exp_k_u = torch.exp(k + u.unsqueeze(0).unsqueeze(0))  # [B, T, C]
        num_bonus = num + exp_k_u * v
        den_bonus = den + exp_k_u
        
        # 应用 receptance gate
        wkv = r * num_bonus / (den_bonus + 1e-6)
        
        return wkv
    
    def wkv_forward_recursive(self, r, k, v, state):
        """
        递推 WKV 计算 (推理时使用)
        只计算一步，维护状态
        
        r, k, v: [batch, 1, dim] - 单步输入
        state: (num, den) - 累积状态
        """
        B, _, C = k.shape
        
        # 提取状态
        if state is None or len(state) == 0:
            num = torch.zeros(B, C, device=k.device)
            den = torch.zeros(B, C, device=k.device)
        else:
            num, den = state
        
        # 参数
        w = -torch.abs(self.time_decay)  # [C]
        u = self.time_bonus               # [C]
        
        # 移除时间维度
        r = r.squeeze(1)  # [B, C]
        k = k.squeeze(1)
        v = v.squeeze(1)
        
        # 计算
        exp_k = torch.exp(k)
        exp_w = torch.exp(w)
        
        # 更新状态
        num_new = exp_k * v + num * exp_w
        den_new = exp_k + den * exp_w
        
        # Bonus
        exp_k_u = torch.exp(k + u)
        num_bonus = num_new + exp_k_u * v
        den_bonus = den_new + exp_k_u
        
        # 输出
        wkv = r * num_bonus / (den_bonus + 1e-6)
        
        return wkv.unsqueeze(1), (num_new, den_new)


class RWKV_ChannelMix(nn.Module):
    """
    RWKV Channel Mixing (类似 FFN，但有 time mixing)
    """
    def __init__(self, dim, layer_id=0):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        
        # Time mixing 参数
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim))
        
        # 投影层
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim * 4, bias=False)  # 扩展 4 倍
        self.value = nn.Linear(dim * 4, dim, bias=False)
        
    def forward(self, x, state=None):
        B, T, C = x.shape
        
        # Time mixing
        if state is not None:
            prev_x = state[0]
            xx = torch.cat([prev_x.unsqueeze(1), x[:, :-1, :]], dim=1)
        else:
            xx = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        
        # Channel mixing
        r = torch.sigmoid(self.receptance(xr))
        k = torch.square(torch.relu(self.key(xk)))  # 平方 ReLU
        v = self.value(k)
        
        out = r * v
        
        return out, x[:, -1, :] if state is not None else None


class RWKVBlock(nn.Module):
    """
    完整的 RWKV Block
    """
    def __init__(self, dim, layer_id=0):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        
        # 两个子层
        self.ln1 = nn.LayerNorm(dim)
        self.time_mixing = RWKV_TimeMix(dim, layer_id)
        
        self.ln2 = nn.LayerNorm(dim)
        self.channel_mixing = RWKV_ChannelMix(dim, layer_id)
        
    def forward(self, x, state=None):
        """
        x: [B, T, D]
        state: tuple of (time_state, channel_state)
        """
        # Time mixing with residual
        time_out, new_time_state = self.time_mixing(self.ln1(x), 
                                                    state[0] if state else None)
        x = x + time_out
        
        # Channel mixing with residual
        channel_out, new_channel_state = self.channel_mixing(self.ln2(x),
                                                               state[1] if state else None)
        x = x + channel_out
        
        return x, (new_time_state, new_channel_state)


class RWKV(nn.Module):
    """
    完整的 RWKV 模型
    """
    def __init__(self, vocab_size, dim=512, n_layers=12, max_seq_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
        # 词嵌入
        self.emb = nn.Embedding(vocab_size, dim)
        
        # RWKV 层
        self.blocks = nn.ModuleList([
            RWKVBlock(dim, i) for i in range(n_layers)
        ])
        
        # 输出层
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, input_ids, state=None):
        """
        input_ids: [B, T]
        state: list of block states for inference
        """
        B, T = input_ids.shape
        
        # 词嵌入
        x = self.emb(input_ids)  # [B, T, D]
        
        # 通过 RWKV blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            block_state = state[i] if state else None
            x, new_state = block(x, block_state)
            new_states.append(new_state)
        
        # 输出
        x = self.ln_out(x)
        logits = self.head(x)
        
        return logits, new_states
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        自回归生成
        """
        self.eval()
        B, T = input_ids.shape
        
        # 初始化状态
        state = [None] * self.n_layers
        
        # 先处理提示词
        logits, state = self.forward(input_ids, state)
        
        # 生成新 token
        for _ in range(max_new_tokens):
            # 取最后一个位置的 logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # Softmax 和采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 更新输入和状态
            logits, state = self.forward(next_token, state)
            
            # 保存结果
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def test_rwkv():
    """测试 RWKV 实现"""
    print("Testing RWKV implementation...")
    
    # 创建小模型
    model = RWKV(vocab_size=1000, dim=64, n_layers=2, max_seq_len=128)
    
    # 测试前向传播
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits, state = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, 1000)")
    
    assert logits.shape == (batch_size, seq_len, 1000), "Output shape mismatch!"
    
    # 测试生成
    print("\nTesting generation...")
    prompt = torch.randint(0, 1000, (1, 5))
    output = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"Prompt length: {prompt.shape[1]}")
    print(f"Output length: {output.shape[1]}")
    print(f"New tokens generated: {output.shape[1] - prompt.shape[1]}")
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_rwkv()
