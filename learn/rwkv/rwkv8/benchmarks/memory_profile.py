"""
内存使用分析工具

分析 RWKV 与 Transformer 在不同序列长度下的内存占用
"""

import torch
import torch.nn as nn
import tracemalloc
import gc
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.rwkv_base import RWKV_TimeMix


class SimpleTransformer(nn.Module):
    """简化版 Transformer 用于测试"""
    def __init__(self, dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim*4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def measure_memory_usage(model, input_shape, device='cuda'):
    """
    测量模型的内存使用
    
    Returns:
        dict: 包含各种内存指标的字典
    """
    if device == 'cuda' and torch.cuda.is_available():
        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始内存
        mem_before = torch.cuda.memory_allocated()
        
        # 创建输入
        B, T, D = input_shape
        x = torch.randn(B, T, D, device=device)
        
        # 前向传播
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            _ = model(x)
        
        # 记录内存
        mem_after = torch.cuda.memory_allocated()
        mem_peak = torch.cuda.max_memory_allocated()
        
        return {
            'memory_increase_mb': (mem_after - mem_before) / 1024**2,
            'peak_memory_mb': mem_peak / 1024**2,
            'memory_before_mb': mem_before / 1024**2,
            'memory_after_mb': mem_after / 1024**2,
        }
    
    else:
        # CPU 内存测量 (使用 tracemalloc)
        tracemalloc.start()
        
        B, T, D = input_shape
        x = torch.randn(B, T, D)
        
        model.eval()
        with torch.no_grad():
            _ = model(x)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'memory_increase_mb': current / 1024**2,
            'peak_memory_mb': peak / 1024**2,
        }


def analyze_memory_scaling():
    """
    分析不同序列长度下的内存使用 scaling
    """
    print("=" * 80)
    print("Memory Scaling Analysis")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 配置
    dim = 512
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    batch_size = 1
    
    results = {
        'rwkv': [],
        'transformer': [],
    }
    
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        
        # RWKV
        try:
            rwkv = RWKV_TimeMix(dim=dim, layer_id=0)
            mem_rwkv = measure_memory_usage(
                rwkv, 
                (batch_size, seq_len, dim),
                device=device
            )
            results['rwkv'].append({
                'seq_len': seq_len,
                **mem_rwkv
            })
            print(f"  RWKV: {mem_rwkv['peak_memory_mb']:.2f} MB")
        except RuntimeError as e:
            print(f"  RWKV: OOM ({str(e)[:50]}...)")
            results['rwkv'].append(None)
        
        # Transformer
        try:
            transformer = SimpleTransformer(dim=dim, num_heads=8, num_layers=2)
            mem_trans = measure_memory_usage(
                transformer,
                (batch_size, seq_len, dim),
                device=device
            )
            results['transformer'].append({
                'seq_len': seq_len,
                **mem_trans
            })
            print(f"  Transformer: {mem_trans['peak_memory_mb']:.2f} MB")
        except RuntimeError as e:
            print(f"  Transformer: OOM ({str(e)[:50]}...)")
            results['transformer'].append(None)
    
    # 分析 scaling
    print("\n" + "=" * 80)
    print("Scaling Analysis")
    print("=" * 80)
    
    # 计算复杂度
    valid_rwkv = [r for r in results['rwkv'] if r is not None]
    valid_trans = [r for r in results['transformer'] if r is not None]
    
    if len(valid_rwkv) >= 2:
        # 计算 RWKV 的 scaling 指数
        seqs = [r['seq_len'] for r in valid_rwkv]
        mems = [r['peak_memory_mb'] for r in valid_rwkv]
        
        # 拟合 log(mem) ~ a * log(seq) + b
        log_seq = np.log(seqs)
        log_mem = np.log(mems)
        
        # 简单线性回归
        A = np.vstack([log_seq, np.ones(len(log_seq))]).T
        slope, intercept = np.linalg.lstsq(A, log_mem, rcond=None)[0]
        
        print(f"\nRWKV Memory Scaling:")
        print(f"  Measured scaling: O(N^{slope:.2f})")
        print(f"  Theoretical: O(N)")
        print(f"  Data points:")
        for s, m in zip(seqs, mems):
            print(f"    Seq={s}: {m:.2f} MB")
    
    if len(valid_trans) >= 2:
        # 计算 Transformer 的 scaling
        seqs = [r['seq_len'] for r in valid_trans]
        mems = [r['peak_memory_mb'] for r in valid_trans]
        
        log_seq = np.log(seqs)
        log_mem = np.log(mems)
        
        A = np.vstack([log_seq, np.ones(len(log_seq))]).T
        slope, intercept = np.linalg.lstsq(A, log_mem, rcond=None)[0]
        
        print(f"\nTransformer Memory Scaling:")
        print(f"  Measured scaling: O(N^{slope:.2f})")
        print(f"  Theoretical: O(N^2)")
        print(f"  Data points:")
        for s, m in zip(seqs, mems):
            print(f"    Seq={s}: {m:.2f} MB")
    
    # 生成可视化
    try:
        generate_plots(results)
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
    
    return results


def generate_plots(results):
    """生成可视化图表"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 准备数据
    valid_rwkv = [r for r in results['rwkv'] if r is not None]
    valid_trans = [r for r in results['transformer'] if r is not None]
    
    # 图 1: 内存使用 vs 序列长度
    ax = axes[0, 0]
    if valid_rwkv:
        seqs = [r['seq_len'] for r in valid_rwkv]
        mems = [r['peak_memory_mb'] for r in valid_rwkv]
        ax.plot(seqs, mems, 'o-', label='RWKV', linewidth=2)
    if valid_trans:
        seqs = [r['seq_len'] for r in valid_trans]
        mems = [r['peak_memory_mb'] for r in valid_trans]
        ax.plot(seqs, mems, 's-', label='Transformer', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图 2: 内存使用 (对数坐标)
    ax = axes[0, 1]
    if valid_rwkv and valid_trans:
        all_seqs = sorted(list(set(
            [r['seq_len'] for r in valid_rwkv] + 
            [r['seq_len'] for r in valid_trans]
        )))
        
        rwkv_mems = []
        trans_mems = []
        for s in all_seqs:
            rwkv_m = [r['peak_memory_mb'] for r in valid_rwkv if r['seq_len'] == s]
            trans_m = [r['peak_memory_mb'] for r in valid_trans if r['seq_len'] == s]
            if rwkv_m:
                rwkv_mems.append(rwkv_m[0])
            if trans_m:
                trans_mems.append(trans_m[0])
        
        if rwkv_mems:
            ax.loglog(all_seqs[:len(rwkv_mems)], rwkv_mems, 'o-', label='RWKV', linewidth=2)
        if trans_mems:
            ax.loglog(all_seqs[:len(trans_mems)], trans_mems, 's-', label='Transformer', linewidth=2)
    
    ax.set_xlabel('Sequence Length (log scale)')
    ax.set_ylabel('Peak Memory (MB, log scale)')
    ax.set_title('Memory Scaling (Log-Log Plot)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # 图 3: 吞吐量对比
    ax = axes[1, 0]
    if valid_rwkv and valid_trans:
        seqs = [r['seq_len'] for r in valid_rwkv if any(t['seq_len'] == r['seq_len'] for t in valid_trans)]
        rwkv_throughput = [r['throughput'] for r in valid_rwkv if r['seq_len'] in seqs]
        trans_throughput = [t['throughput'] for t in valid_trans if t['seq_len'] in seqs]
        
        x = np.arange(len(seqs))
        width = 0.35
        
        ax.bar(x - width/2, rwkv_throughput, width, label='RWKV', alpha=0.8)
        ax.bar(x + width/2, trans_throughput, width, label='Transformer', alpha=0.8)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Throughput (tokens/sec)')
        ax.set_title('Throughput Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(seqs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 图 4: 速度提升倍数
    ax = axes[1, 1]
    if valid_rwkv and valid_trans:
        common_seqs = []
        speedups = []
        
        for rwkv_r in valid_rwkv:
            seq = rwkv_r['seq_len']
            trans_r = next((t for t in valid_trans if t['seq_len'] == seq), None)
            if trans_r:
                speedup = trans_r['time_mean'] / rwkv_r['time_mean']
                common_seqs.append(seq)
                speedups.append(speedup)
        
        if speedups:
            colors = ['green' if s > 1 else 'red' for s in speedups]
            ax.bar(common_seqs, speedups, color=colors, alpha=0.7)
            ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Equal speed')
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Speedup (Trans Time / RWKV Time)')
            ax.set_title('RWKV Speedup over Transformer')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to benchmark_results.png")
    
    plt.show()


if __name__ == "__main__":
    results = analyze_memory_scaling()
