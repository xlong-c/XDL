# -*- coding: utf-8 -*-
"""
Multi-Head Latent Attention (MLA) 实现
基于 DeepSeek 的 MLA 架构

MLA的核心思想:
1. 使用低维度的 latent key 和 latent value 来压缩 KV cache
2. 只存储压缩后的 KV，大幅减少内存占用
3. 通过 up-projection 恢复到所需的维度进行注意力计算

内存优化:
- MHA: O(seq_len * num_heads * head_dim * 2)
- MLA: O(seq_len * latent_dim * 2)，其中 latent_dim << num_heads * head_dim

典型配置:
- hidden_dim = 4096
- num_heads = 32
- head_dim = 128 (total = 4096)
- latent_dim = 512 (压缩8倍)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadLatentAttention(nn.Module):
    """
    多头潜在注意力 (Multi-Head Latent Attention)

    核心机制:
    1. 压缩 Key/Value 到低维 latent space
    2. 只存储压缩后的 KV (大幅减少缓存)
    3. 使用 up-projection 恢复维度进行注意力计算

    参数:
        hidden_dim: 模型隐藏层维度
        num_heads: 注意力头数量
        head_dim: 每个注意力头的维度
        latent_dim: 压缩后的 KV 维度 (关键参数，决定压缩率)
        dropout: dropout概率
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        latent_dim: int = 64,  # 压缩维度，通常远小于 num_heads * head_dim
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # 计算 head_dim
        if head_dim is None:
            head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim

        assert self.num_heads * self.head_dim == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Query 投影 (不压缩，保持完整维度)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Key/Value 压缩投影: hidden_dim -> latent_dim
        self.W_down_k = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.W_down_v = nn.Linear(hidden_dim, latent_dim, bias=False)

        # Key/Value 上投影: latent_dim -> num_heads * head_dim
        # 注意：这里是共享的上投影权重，用于所有位置
        output_dim = num_heads * self.head_dim
        self.W_up_k = nn.Linear(latent_dim, output_dim, bias=False)
        self.W_up_v = nn.Linear(latent_dim, output_dim, bias=False)

        # 输出投影
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # KV Cache - 只存储压缩后的 latent KV
        self.cache_kv: Optional[torch.Tensor] = None  # [batch_size, seq_len, latent_dim * 2]

    def compress_kv(
        self,
        k_latent: torch.Tensor,
        v_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        压缩 KV: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, latent_dim * 2]

        将多头 KV 压缩到 latent space，用于缓存

        参数:
            k_latent: 压缩后的 latent K [batch_size, seq_len, latent_dim]
            v_latent: 压缩后的 latent V [batch_size, seq_len, latent_dim]

        返回:
            压缩后的 KV 张量 [batch_size, seq_len, latent_dim * 2]
        """
        # 拼接 K 和 V 的 latent 表示
        batch_size, seq_len, _ = k_latent.shape
        kv_compressed = torch.cat([k_latent, v_latent], dim=-1)  # [batch_size, seq_len, latent_dim * 2]
        return kv_compressed

    def decompress_kv(
        self,
        kv_compressed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解压 KV: [batch_size, seq_len, latent_dim * 2] -> [batch_size, num_heads, seq_len, head_dim]

        从 latent space 恢复到多头维度用于注意力计算

        参数:
            kv_compressed: 压缩的 KV 张量 [batch_size, seq_len, latent_dim * 2]

        返回:
            k: 上投影后的 K [batch_size, num_heads, seq_len, head_dim]
            v: 上投影后的 V [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = kv_compressed.shape

        # 分离 K 和 V 的 latent 表示
        k_latent = kv_compressed[:, :, :self.latent_dim]  # [batch_size, seq_len, latent_dim]
        v_latent = kv_compressed[:, :, self.latent_dim:]  # [batch_size, seq_len, latent_dim]

        # 上投影到多头维度
        k_up = self.W_up_k(k_latent)  # [batch_size, seq_len, num_heads * head_dim]
        v_up = self.W_up_v(v_latent)  # [batch_size, seq_len, num_heads * head_dim]

        # 重塑为多头格式
        k = k_up.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_up.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len, head_dim]

        return k, v

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        past_key_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len, seq_len]
            use_cache: 是否使用 KV Cache
            past_key_value: 过去的压缩 KV 缓存 [batch_size, seq_len, latent_dim * 2]

        返回:
            output: 注意力输出 [batch_size, seq_len, hidden_dim]
            present_key_value: 当前的压缩 KV 缓存
        """
        batch_size, seq_len, _ = x.shape

        # 1. 计算 Q (不压缩)
        q = self.W_q(x)  # [batch_size, seq_len, hidden_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len, head_dim]

        # 2. 压缩 Key 和 Value 到 latent space
        k_latent = self.W_down_k(x)  # [batch_size, seq_len, latent_dim]
        v_latent = self.W_down_v(x)  # [batch_size, seq_len, latent_dim]

        # 3. 压缩 KV 用于缓存
        kv_compressed = self.compress_kv(k_latent, v_latent)  # [batch_size, seq_len, latent_dim * 2]

        # 4. 处理 KV Cache
        if past_key_value is not None:
            # 拼接历史缓存 (压缩的 KV)
            kv_compressed = torch.cat([past_key_value, kv_compressed], dim=1)

        # 更新内部缓存
        if use_cache:
            self.cache_kv = kv_compressed
            present_key_value = kv_compressed
        else:
            present_key_value = None

        # 5. 解压 KV 用于注意力计算
        k, v = self.decompress_kv(kv_compressed)
        # [batch_size, num_heads, total_seq_len, head_dim]

        # 6. 计算注意力分数
        # scores: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 7. 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # 8. Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 9. 加权求和
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]

        # 10. 拼接多头并投影
        output = output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        output = output.view(batch_size, seq_len, self.hidden_dim)  # [batch_size, seq_len, hidden_dim]
        output = self.W_o(output)

        return output, present_key_value

    def get_kv_cache_size(self) -> int:
        """
        获取当前 KV Cache 的内存占用（字节数）

        Returns:
            KV Cache 占用的字节数
        """
        if self.cache_kv is None:
            return 0
        # 计算压缩后的 KV 张量大小
        return self.cache_kv.numel() * 4  # float32 = 4 bytes

    def get_cache_seq_len(self) -> int:
        """获取当前缓存的序列长度"""
        if self.cache_kv is None:
            return 0
        return self.cache_kv.shape[1]

    def get_compression_ratio(self) -> float:
        """
        获取压缩率

        Returns:
            压缩率 = (num_heads * head_dim) / latent_dim
        """
        return (self.num_heads * self.head_dim) / self.latent_dim

    def reset_cache(self):
        """重置 KV Cache"""
        self.cache_kv = None


def test_mla():
    """测试 MLA 基本功能"""
    print("=== 测试 Multi-Head Latent Attention ===\n")

    # 创建 MLA 层
    mla = MultiHeadLatentAttention(
        hidden_dim=512,
        num_heads=8,
        head_dim=64,
        latent_dim=64,  # 压缩到 1/8
    )

    # 测试输入
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 512)

    print(f"输入形状: {x.shape}")
    print(f"隐藏层维度: {mla.hidden_dim}")
    print(f"注意力头数量: {mla.num_heads}")
    print(f"每个头的维度: {mla.head_dim}")
    print(f"压缩维度 (latent_dim): {mla.latent_dim}")
    print(f"压缩率: {mla.get_compression_ratio():.1f}x\n")

    # 1. 前向传播（无缓存）
    output, _ = mla(x, use_cache=False)
    print(f"输出形状 (无缓存): {output.shape}")
    assert output.shape == x.shape, "输出形状应该与输入相同"
    print("✓ 无缓存前向传播成功\n")

    # 2. 测试 KV Cache 机制
    mla.reset_cache()

    # 第一个 token
    token1 = x[:, :1, :]
    output1, cache1 = mla(token1, use_cache=True)
    print(f"第1个token后 - 缓存长度: {mla.get_cache_seq_len()}")
    print(f"第1个token后 - KV Cache大小: {mla.get_kv_cache_size() / 1024:.2f} KB")

    # 第二个 token（复用缓存）
    token2 = x[:, 1:2, :]
    output2, cache2 = mla(token2, use_cache=True, past_key_value=cache1)
    print(f"第2个token后 - 缓存长度: {mla.get_cache_seq_len()}")
    print(f"第2个token后 - KV Cache大小: {mla.get_kv_cache_size() / 1024:.2f} KB")

    # 继续添加 token
    for i in range(2, seq_len):
        token = x[:, i:i+1, :]
        output, cache2 = mla(token, use_cache=True, past_key_value=cache2)

    print(f"最终缓存长度: {mla.get_cache_seq_len()}")
    print(f"最终KV Cache大小: {mla.get_kv_cache_size() / 1024:.2f} KB")
    print("✓ KV Cache 机制测试成功\n")

    # 3. 计算内存占用分析
    seq_len = 4096
    hidden_dim = 4096
    num_heads = 32
    head_dim = 128
    latent_dim = 512  # 压缩到 1/8

    print("=== 长序列内存占用分析 ===")
    print(f"序列长度: {seq_len}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"注意力头数量: {num_heads}")
    print(f"每个头的维度: {head_dim}")
    print(f"压缩维度: {latent_dim}")
    print(f"压缩率: {num_heads * head_dim / latent_dim:.1f}x\n")

    # MHA 的 KV Cache 大小
    mha_cache_k = torch.randn(1, num_heads, seq_len, head_dim)
    mha_cache_v = torch.randn(1, num_heads, seq_len, head_dim)
    mha_bytes = (mha_cache_k.numel() + mha_cache_v.numel()) * 4

    # MLA 的 KV Cache 大小
    mla_cache = torch.randn(1, seq_len, latent_dim * 2)
    mla_bytes = mla_cache.numel() * 4

    print(f"MHA KV Cache总大小: {mha_bytes / 1024 / 1024:.2f} MB")
    print(f"MLA KV Cache总大小: {mla_bytes / 1024 / 1024:.2f} MB")
    print(f"内存节省: {(1 - mla_bytes / mha_bytes) * 100:.1f}%")
    print(f"MHA单个token增加的缓存: {mha_bytes / seq_len / 1024:.2f} KB/token")
    print(f"MLA单个token增加的缓存: {mla_bytes / seq_len / 1024:.2f} KB/token")
    print("✓ MLA 测试完成\n")

    return mla


if __name__ == "__main__":
    test_mla()
