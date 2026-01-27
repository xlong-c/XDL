# -*- coding: utf-8 -*-
"""
MHA vs MLA 对比实验
展示 Multi-Head Attention 和 Multi-Head Latent Attention 的差异

对比维度:
1. 内存占用 (KV Cache 大小)
2. 计算复杂度
3. 序列长度扩展的影响
4. 不同压缩率的效果
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

from mha import MultiHeadAttention
from mla import MultiHeadLatentAttention

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def compare_memory_usage():
    """对比不同序列长度下的内存占用"""
    print("=" * 70)
    print("实验 1: 内存占用对比")
    print("=" * 70)

    # 模型配置 (类似 DeepSeek-V2)
    hidden_dim = 5120
    num_heads = 40
    head_dim = hidden_dim // num_heads  # 128
    latent_dim = 512  # 压缩率 10x

    # 创建模型
    mha = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim)
    mla = MultiHeadLatentAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, latent_dim=latent_dim)

    print(f"\n模型配置:")
    print(f"  隐藏层维度: {hidden_dim}")
    print(f"  注意力头数量: {num_heads}")
    print(f"  每个头的维度: {head_dim}")
    print(f"  MLA 压缩维度: {latent_dim}")
    print(f"  压缩率: {num_heads * head_dim / latent_dim:.1f}x\n")

    # 测试不同序列长度
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    mha_memory = []
    mla_memory = []

    print(f"{'序列长度':<12} {'MHA (MB)':<12} {'MLA (MB)':<12} {'节省':<10}")
    print("-" * 50)

    for seq_len in seq_lengths:
        # 计算内存占用
        mha_cache = torch.randn(1, num_heads, seq_len, head_dim)
        mha_cache_v = torch.randn(1, num_heads, seq_len, head_dim)
        mha_mem = (mha_cache.numel() + mha_cache_v.numel()) * 4 / (1024 ** 2)

        mla_cache = torch.randn(1, seq_len, latent_dim * 2)
        mla_mem = mla_cache.numel() * 4 / (1024 ** 2)

        mha_memory.append(mha_mem)
        mla_memory.append(mla_mem)

        savings = (1 - mla_mem / mha_mem) * 100
        print(f"{seq_len:<12} {mha_mem:<12.2f} {mla_mem:<12.2f} {savings:<10.1f}%")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 内存占用对比
    ax1.plot(seq_lengths, mha_memory, 'b-o', label='MHA', linewidth=2, markersize=8)
    ax1.plot(seq_lengths, mla_memory, 'r-s', label='MLA', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('KV Cache Memory (MB)', fontsize=12)
    ax1.set_title('KV Cache Memory Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 内存节省率
    savings = [(1 - mla_mem / mha_mem) * 100 for mha_mem, mla_mem in zip(mha_memory, mla_memory)]
    ax2.bar(range(len(seq_lengths)), savings, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Memory Savings (%)', fontsize=12)
    ax2.set_title('MLA Memory Savings', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(seq_lengths)))
    ax2.set_xticklabels([str(l) for l in seq_lengths], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'memory_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison chart saved: results/memory_comparison.png")
    plt.show()


def compare_compression_ratios():
    """对比不同压缩率的效果"""
    print("\n" + "=" * 70)
    print("实验 2: 不同压缩率的效果")
    print("=" * 70)

    # 固定配置
    hidden_dim = 5120
    num_heads = 40
    head_dim = hidden_dim // num_heads
    seq_len = 4096

    # 不同的压缩率
    compression_ratios = [2, 4, 6, 8, 10, 12, 16]
    latent_dims = [int((num_heads * head_dim) / ratio) for ratio in compression_ratios]

    print(f"\n序列长度: {seq_len}")
    print(f"隐藏层维度: {hidden_dim}\n")

    print(f"{'压缩率':<10} {'Latent Dim':<12} {'MHA (MB)':<12} {'MLA (MB)':<12} {'节省':<10}")
    print("-" * 60)

    # 计算 MHA 基准
    mha_cache = torch.randn(1, num_heads, seq_len, head_dim)
    mha_cache_v = torch.randn(1, num_heads, seq_len, head_dim)
    mha_mem = (mha_cache.numel() + mha_cache_v.numel()) * 4 / (1024 ** 2)

    mla_memories = []

    for ratio, latent_dim in zip(compression_ratios, latent_dims):
        mla_cache = torch.randn(1, seq_len, latent_dim * 2)
        mla_mem = mla_cache.numel() * 4 / (1024 ** 2)
        mla_memories.append(mla_mem)

        savings = (1 - mla_mem / mha_mem) * 100
        print(f"{ratio:<10} {latent_dim:<12} {mha_mem:<12.2f} {mla_mem:<12.2f} {savings:<10.1f}%")

    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(compression_ratios, mla_memories, 'g-o', linewidth=2, markersize=8, label='MLA')
    ax.axhline(y=mha_mem, color='b', linestyle='--', linewidth=2, label='MHA (Baseline)')

    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('KV Cache Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage at Different Compression Ratios', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'compression_ratio_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Compression ratio chart saved: results/compression_ratio_analysis.png")
    plt.show()


def compare_inference_speed():
    """对比推理速度"""
    print("\n" + "=" * 70)
    print("实验 3: 推理速度对比")
    print("=" * 70)

    # 模型配置
    hidden_dim = 512
    num_heads = 8
    head_dim = hidden_dim // num_heads
    latent_dim = 64
    batch_size = 4
    seq_len = 128

    # 创建模型
    mha = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim)
    mla = MultiHeadLatentAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, latent_dim=latent_dim)

    # 预热
    x = torch.randn(batch_size, seq_len, hidden_dim)
    with torch.no_grad():
        mha(x, use_cache=False)
        mla(x, use_cache=False)

    # 测试前向传播速度
    num_iterations = 100

    # MHA 速度
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            mha(x, use_cache=False)
    mha_time = (time.time() - start_time) / num_iterations * 1000  # ms

    # MLA 速度
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            mla(x, use_cache=False)
    mla_time = (time.time() - start_time) / num_iterations * 1000  # ms

    print(f"\n模型配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  隐藏层维度: {hidden_dim}")
    print(f"  注意力头数量: {num_heads}")
    print(f"  压缩维度: {latent_dim}\n")

    print(f"推理速度 (前向传播):")
    print(f"  MHA: {mha_time:.2f} ms")
    print(f"  MLA: {mla_time:.2f} ms")
    print(f"  差异: {((mla_time - mha_time) / mha_time * 100):+.1f}%")
    print(f"\n✓ 推理速度对比完成")


def simulate_autoregressive_generation():
    """模拟自回归生成过程，展示 KV Cache 的累积效果"""
    print("\n" + "=" * 70)
    print("实验 4: 自回归生成中的 KV Cache 累积")
    print("=" * 70)

    # 模型配置
    hidden_dim = 2048
    num_heads = 16
    head_dim = hidden_dim // num_heads
    latent_dim = 256

    mha = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim)
    mla = MultiHeadLatentAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, latent_dim=latent_dim)

    print(f"\n模型配置:")
    print(f"  隐藏层维度: {hidden_dim}")
    print(f"  注意力头数量: {num_heads}")
    print(f"  压缩维度: {latent_dim}")
    print(f"  压缩率: {num_heads * head_dim / latent_dim:.1f}x\n")

    # 模拟生成过程
    max_tokens = 128
    mha_cache_sizes = []
    mla_cache_sizes = []

    mha.reset_cache()
    mla.reset_cache()

    print(f"{'Token':<10} {'MHA Cache (KB)':<18} {'MLA Cache (KB)':<18} {'累积节省':<12}")
    print("-" * 65)

    # 初始化缓存变量
    mha_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    mla_cache: Optional[torch.Tensor] = None

    for i in range(1, max_tokens + 1):
        # 模拟单个 token 的输入
        token = torch.randn(1, 1, hidden_dim)

        # MHA
        if i == 1:
            _, mha_cache = mha(token, use_cache=True)
        else:
            _, mha_cache = mha(token, use_cache=True, past_key_value=mha_cache)
        mha_cache_sizes.append(mha.get_kv_cache_size() / 1024)

        # MLA
        if i == 1:
            _, mla_cache = mla(token, use_cache=True)
        else:
            _, mla_cache = mla(token, use_cache=True, past_key_value=mla_cache)
        mla_cache_sizes.append(mla.get_kv_cache_size() / 1024)

        if i % 16 == 0 or i == max_tokens:
            savings = (1 - mla_cache_sizes[-1] / mha_cache_sizes[-1]) * 100
            print(f"{i:<10} {mha_cache_sizes[-1]:<18.2f} {mla_cache_sizes[-1]:<18.2f} {savings:<12.1f}%")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # KV Cache 大小累积
    ax1.plot(range(1, max_tokens + 1), mha_cache_sizes, 'b-o', label='MHA', linewidth=2, markersize=4)
    ax1.plot(range(1, max_tokens + 1), mla_cache_sizes, 'r-s', label='MLA', linewidth=2, markersize=4)
    ax1.set_xlabel('Number of Generated Tokens', fontsize=12)
    ax1.set_ylabel('KV Cache Size (KB)', fontsize=12)
    ax1.set_title('KV Cache Accumulation in Autoregressive Generation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 单个 token 增加的缓存
    mha_per_token = np.diff(mha_cache_sizes, prepend=0)
    mla_per_token = np.diff(mla_cache_sizes, prepend=0)

    ax2.bar(['MHA', 'MLA'], [mha_per_token[0], mla_per_token[0]], color=['steelblue', 'indianred'], alpha=0.7)
    ax2.set_ylabel('KV Cache per Token (KB)', fontsize=12)
    ax2.set_title('KV Cache Added per Token', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    ax2.text(0, mha_per_token[0], f'{mha_per_token[0]:.2f}', ha='center', va='bottom', fontsize=10)
    ax2.text(1, mla_per_token[0], f'{mla_per_token[0]:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'autoregressive_cache.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Autoregressive generation chart saved: results/autoregressive_cache.png")
    plt.show()


def summary():
    """总结报告"""
    print("\n" + "=" * 70)
    print("总结报告: MHA vs MLA")
    print("=" * 70)

    print("\n核心差异:")
    print("1. MHA (Multi-Head Attention):")
    print("   - 存储完整的 K、V 矩阵")
    print("   - KV Cache 大小: O(seq_len * num_heads * head_dim * 2)")
    print("   - 优点: 实现简单，无额外计算开销")
    print("   - 缺点: 长序列时内存占用巨大")

    print("\n2. MLA (Multi-Head Latent Attention):")
    print("   - 将 K、V 压缩到低维 latent space")
    print("   - KV Cache 大小: O(seq_len * latent_dim * 2)")
    print("   - 优点: 大幅减少内存占用 (通常节省 80-90%)")
    print("   - 缺点: 增加了上投影的计算开销")

    print("\n适用场景:")
    print("- MHA: 短序列、对推理速度要求极高的场景")
    print("- MLA: 长序列、内存受限的场景 (如大语言模型)")

    print("\nDeepSeek 的选择:")
    print("- DeepSeek-V2/V3 使用 MLA 来支持超长上下文")
    print("- 压缩率通常设置为 8-12 倍")
    print("- 在性能和效率之间取得了良好平衡")


def main():
    """运行所有对比实验"""
    print("\n" + "=" * 70)
    print("MHA vs MLA 对比实验套件")
    print("=" * 70)

    # 运行各个实验
    compare_memory_usage()
    compare_compression_ratios()
    compare_inference_speed()
    simulate_autoregressive_generation()
    summary()

    print("\n" + "=" * 70)
    print("所有实验完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
