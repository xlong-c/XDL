# -*- coding: utf-8 -*-
"""
Multi-Head Attention (MHA) 实现作为对比基线
基于标准的注意力机制实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    标准多头注意力机制 (Multi-Head Attention)

    标准MHA的特点:
    1. 每个头有独立的Q、K、V投影
    2. KV Cache存储完整的K、V矩阵
    3. 内存占用: O(seq_len * num_heads * head_dim * 2)

    参数:
        hidden_dim: 模型隐藏层维度
        num_heads: 注意力头数量
        head_dim: 每个注意力头的维度 (通常 = hidden_dim // num_heads)
        dropout: dropout概率
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_heads
        assert self.num_heads * self.head_dim == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Q、K、V投影权重 (标准的MHA每个头都有独立的投影，但这里统一实现)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 输出投影
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # KV Cache
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len, seq_len]
            use_cache: 是否使用KV Cache
            past_key_value: 过去的KV缓存 [k_cache, v_cache]

        返回:
            output: 注意力输出 [batch_size, seq_len, hidden_dim]
            present_key_value: 当前的KV缓存 (如果use_cache=True)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 计算 Q、K、V
        q = self.W_q(x)  # [batch_size, seq_len, hidden_dim]
        k = self.W_k(x)  # [batch_size, seq_len, hidden_dim]
        v = self.W_v(x)  # [batch_size, seq_len, hidden_dim]

        # 2. 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在形状: [batch_size, num_heads, seq_len, head_dim]

        # 3. 处理 KV Cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # 拼接历史缓存
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # 更新内部缓存
        if use_cache:
            self.cache_k = k
            self.cache_v = v
            present_key_value = (k, v)
        else:
            present_key_value = None

        # 4. 计算注意力分数
        # scores: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 5. 应用注意力掩码
        if attention_mask is not None:
            # 将掩码扩展到多头维度
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # 6. Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. 加权求和
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]

        # 8. 拼接多头并投影
        output = output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        output = output.view(batch_size, seq_len, self.hidden_dim)  # [batch_size, seq_len, hidden_dim]
        output = self.W_o(output)

        return output, present_key_value

    def get_kv_cache_size(self) -> int:
        """
        获取当前KV Cache的内存占用（字节数）

        Returns:
            KV Cache占用的字节数
        """
        if self.cache_k is None or self.cache_v is None:
            return 0
        # 计算张量大小: num_elements * element_size (float32 = 4 bytes)
        k_size = self.cache_k.numel() * 4  # K cache
        v_size = self.cache_v.numel() * 4  # V cache
        return k_size + v_size

    def get_cache_seq_len(self) -> int:
        """获取当前缓存的序列长度"""
        if self.cache_k is None:
            return 0
        return self.cache_k.shape[2]

    def reset_cache(self):
        """重置KV Cache"""
        self.cache_k = None
        self.cache_v = None


def test_mha():
    """测试MHA基本功能"""
    print("=== 测试 Multi-Head Attention ===\n")

    # 创建MHA层
    mha = MultiHeadAttention(hidden_dim=512, num_heads=8, head_dim=64)

    # 测试输入
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 512)

    print(f"输入形状: {x.shape}")
    print(f"隐藏层维度: {mha.hidden_dim}")
    print(f"注意力头数量: {mha.num_heads}")
    print(f"每个头的维度: {mha.head_dim}\n")

    # 1. 前向传播（无缓存）
    output, _ = mha(x, use_cache=False)
    print(f"输出形状 (无缓存): {output.shape}")
    assert output.shape == x.shape, "输出形状应该与输入相同"
    print("✓ 无缓存前向传播成功\n")

    # 2. 测试 KV Cache 机制
    mha.reset_cache()

    # 第一个token
    token1 = x[:, :1, :]
    output1, cache1 = mha(token1, use_cache=True)
    print(f"第1个token后 - 缓存长度: {mha.get_cache_seq_len()}")
    print(f"第1个token后 - KV Cache大小: {mha.get_kv_cache_size() / 1024:.2f} KB")

    # 第二个token（复用缓存）
    token2 = x[:, 1:2, :]
    output2, cache2 = mha(token2, use_cache=True, past_key_value=cache1)
    print(f"第2个token后 - 缓存长度: {mha.get_cache_seq_len()}")
    print(f"第2个token后 - KV Cache大小: {mha.get_kv_cache_size() / 1024:.2f} KB")

    # 继续添加token
    for i in range(2, seq_len):
        token = x[:, i:i+1, :]
        output, cache2 = mha(token, use_cache=True, past_key_value=cache2)

    print(f"最终缓存长度: {mha.get_cache_seq_len()}")
    print(f"最终KV Cache大小: {mha.get_kv_cache_size() / 1024:.2f} KB")
    print("✓ KV Cache 机制测试成功\n")

    # 3. 计算内存占用分析
    seq_len = 4096
    hidden_dim = 4096
    num_heads = 32
    head_dim = 128

    # 模拟长序列的KV Cache大小
    cache_k = torch.randn(1, num_heads, seq_len, head_dim)
    cache_v = torch.randn(1, num_heads, seq_len, head_dim)
    total_bytes = (cache_k.numel() + cache_v.numel()) * 4

    print("=== 长序列内存占用分析 ===")
    print(f"序列长度: {seq_len}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"注意力头数量: {num_heads}")
    print(f"每个头的维度: {head_dim}")
    print(f"KV Cache总大小: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"单个token增加的缓存: {total_bytes / seq_len / 1024:.2f} KB/token")
    print("✓ MHA测试完成\n")

    return mha


if __name__ == "__main__":
    test_mha()
