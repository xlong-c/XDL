# -*- coding: utf-8 -*-
"""
简化的Attention Sink现象演示
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def simulate_attention_sink():
    """模拟attention sink现象"""
    print("=== Attention Sink 现象模拟 ===")

    # 模拟参数
    seq_len = 128
    d_model = 64

    # 设置随机种子
    torch.manual_seed(42)

    # 生成序列(第一个token是特殊的sink token)
    sequence = torch.randn(seq_len, d_model)

    # 让第一个token成为sink token：给它一个显著的embedding
    sequence[0] = torch.zeros(d_model)
    sequence[0, 0] = 5.0  # 显著的值使其更容易被关注

    print(f"序列长度: {seq_len}")
    print(f"第一个token (sink token): [{sequence[0][:5].tolist()}...]")  # 只显示前5个值

    # 简化的注意力计算
    def compute_attention(Q, K):
        """计算注意力权重"""
        scores = torch.matmul(Q, K.T) / np.sqrt(d_model)
        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights

    # 计算注意力
    attention_weights = compute_attention(sequence, sequence)

    # 分析attention sink现象
    first_token_received = attention_weights[:, 0].mean().item()
    other_tokens_avg = attention_weights[:, 1:].mean().item()
    sink_ratio = first_token_received / other_tokens_avg

    print("\n=== Attention Sink 分析 ===")
    print(f"Sink token收到的平均注意力权重: {first_token_received:.4f}")
    print(f"其他token的平均注意力权重: {other_tokens_avg:.4f}")
    print(f"Sink Ratio: {sink_ratio:.2f}")

    if sink_ratio > 2.0:
        print("⚠️  检测到明显的attention sink现象！")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 注意力热图
    im1 = ax1.imshow(attention_weights.numpy(), cmap='Blues', aspect='auto')
    ax1.set_title('Attention Weights Heatmap')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    ax1.axvline(x=0, color='red', linestyle='--', label='Sink Token')
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='Attention Weight')

    # 每个位置收到的总注意力
    total_attention_per_position = attention_weights.sum(dim=0)
    positions = range(seq_len)

    ax2.bar(positions[:20], total_attention_per_position[:20])  # 只显示前20个位置
    ax2.set_title('Attention Received by Each Position (First 20)')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Total Attention Received')
    ax2.axvline(x=0, color='red', linestyle='--', label='Sink Token')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return attention_weights

def compare_with_without_sink():
    """比较有sink token和无sink token的情况"""
    print("\n=== 比较有/无Sink Token的效果 ===")

    seq_len = 64
    d_model = 64
    torch.manual_seed(42)

    # 1. 有sink token的序列
    seq_with_sink = torch.randn(seq_len + 1, d_model)
    seq_with_sink[0] = torch.zeros(d_model)
    seq_with_sink[0, 0] = 5.0  # sink token

    # 2. 无sink token的序列
    seq_without_sink = torch.randn(seq_len, d_model)

    def compute_attention(Q, K):
        scores = torch.matmul(Q, K.T) / np.sqrt(d_model)
        return F.softmax(scores, dim=-1)

    # 计算注意力
    att_with_sink = compute_attention(seq_with_sink, seq_with_sink)
    att_without_sink = compute_attention(seq_without_sink, seq_without_sink)

    # 分析对比
    first_pos_with_sink = att_with_sink[:, 0].mean().item()
    first_pos_without_sink = att_without_sink[:, 0].mean().item()

    print(f"有sink token时第一个位置注意力: {first_pos_with_sink:.4f}")
    print(f"无sink token时第一个位置注意力: {first_pos_without_sink:.4f}")
    print(f"差异: {first_pos_with_sink - first_pos_without_sink:.4f}")

    # 可视化对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 有sink token
    im1 = ax1.imshow(att_with_sink.numpy(), cmap='Blues', aspect='auto')
    ax1.set_title('With Sink Token')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    ax1.axvline(x=0, color='red', linestyle='--')
    ax1.axhline(y=0, color='red', linestyle='--')

    # 无sink token
    im2 = ax2.imshow(att_without_sink.numpy(), cmap='Blues', aspect='auto')
    ax2.set_title('Without Sink Token')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')

    plt.colorbar(im1, ax=ax1, label='Attention Weight')
    plt.colorbar(im2, ax=ax2, label='Attention Weight')

    plt.tight_layout()
    plt.show()

def demonstrate_sequence_length_effect():
    """演示序列长度对attention sink的影响"""
    print("\n=== 序列长度对Attention Sink的影响 ===")

    d_model = 64
    seq_lengths = [16, 32, 64, 128, 256]
    sink_ratios = []

    torch.manual_seed(42)

    for seq_len in seq_lengths:
        # 生成序列(第一个token是sink)
        sequence = torch.randn(seq_len, d_model)
        sequence[0] = torch.zeros(d_model)
        sequence[0, 0] = 5.0

        # 计算注意力
        scores = torch.matmul(sequence, sequence.T) / np.sqrt(d_model)
        attention = F.softmax(scores, dim=-1)

        # 计算sink ratio
        first_token_att = attention[:, 0].mean().item()
        other_tokens_att = attention[:, 1:].mean().item()
        sink_ratio = first_token_att / other_tokens_att

        sink_ratios.append(sink_ratio)
        print(f"序列长度 {seq_len:3d}: Sink Ratio = {sink_ratio:.2f}")

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, sink_ratios, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length')
    plt.ylabel('Sink Ratio')
    plt.title('Attention Sink Intensity vs Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.show()

    return seq_lengths, sink_ratios

def main():
    """主函数"""
    print("Attention Sink 现象演示")
    print("什么是Attention Sink?")
    print("- Transformer模型中的初始token(如[CLS]、[SOS])会获得不成比例的注意力权重")
    print("- 这种现象会影响模型性能和可解释性")
    print("- 在长序列处理中尤其明显\n")

    # 1. 基础attention sink演示
    simulate_attention_sink()

    # 2. 比较有/无sink token
    compare_with_without_sink()

    # 3. 序列长度影响
    seq_lengths, sink_ratios = demonstrate_sequence_length_effect()

    print("\n=== 总结 ===")
    print("Attention Sink的主要原因:")
    print("1. 初始token的特殊嵌入(如[CLS] token)")
    print("2. 训练过程中学习到的模式")
    print("3. 位置编码的影响")
    print("4. 注意力机制的 softmax 特性")
    print("\n缓解方法:")
    print("1. 移除不必要的特殊token")
    print("2. 添加注意力偏置")
    print("3. 修改训练策略")
    print("4. 使用更平衡的位置编码")

if __name__ == "__main__":
    main()