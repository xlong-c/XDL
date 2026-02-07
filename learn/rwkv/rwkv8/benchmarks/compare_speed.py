"""
RWKV 性能对比测试

对比不同实现的性能：
1. PyTorch 原生实现
2. CUDA 优化实现
3. 与 Transformer Attention 的对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, List, Tuple
import json
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.rwkv_base import RWKV_TimeMix


class NativeAttention(nn.Module):
    """标准 Transformer Attention (用于对比)"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v  # [B, H, T, D]
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        
        return out


def benchmark_module(module, input_shape, n_warmup=10, n_iters=100, device='cuda'):
    """
    基准测试模块
    
    Args:
        module: 要测试的 nn.Module
        input_shape: (batch, seq_len, dim)
        n_warmup: 预热迭代次数
        n_iters: 测试迭代次数
        device: 运行设备
    
    Returns:
        dict: 包含各种性能指标的字典
    """
    module = module.to(device)
    module.eval()
    
    B, T, D = input_shape
    x = torch.randn(B, T, D, device=device)
    
    # 预热
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = module(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 测试前向传播
    times_forward = []
    with torch.no_grad():
        for _ in range(n_iters):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            _ = module(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            
            times_forward.append((end - start) * 1000)  # ms
    
    # 测试显存使用
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = module(x)
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        memory_allocated = 0
    
    # 计算 FLOPs (近似)
    if isinstance(module, RWKV_TimeMix):
        # WKV: 每个位置需要 exp, mul, add 等操作
        # 粗略估计: ~10 * B * T * C FLOPs
        flops = 10 * B * T * D
    elif isinstance(module, NativeAttention):
        # Attention: Q@K^T, softmax, @V
        # ~4 * B * T^2 * D FLOPs (主要 bottleneck 在 T^2)
        flops = 4 * B * T * T * D
    else:
        flops = 0
    
    return {
        'time_mean': np.mean(times_forward),
        'time_std': np.std(times_forward),
        'time_min': np.min(times_forward),
        'time_max': np.max(times_forward),
        'memory_mb': memory_allocated,
        'flops': flops,
        'throughput': (B * T) / (np.mean(times_forward) / 1000),  # tokens/sec
    }


def run_comparison(configs: List[Dict], device='cuda'):
    """
    运行对比测试
    
    Args:
        configs: 配置列表，每个配置包含 name, module, input_shape
        device: 运行设备
    
    Returns:
        DataFrame: 结果表格
    """
    import pandas as pd
    
    results = []
    
    for config in configs:
        name = config['name']
        module = config['module']
        input_shape = config['input_shape']
        
        print(f"\nTesting {name}...")
        print(f"  Input shape: {input_shape}")
        
        result = benchmark_module(module, input_shape, device=device)
        result['name'] = name
        result['input_shape'] = str(input_shape)
        results.append(result)
        
        print(f"  Time: {result['time_mean']:.3f} ± {result['time_std']:.3f} ms")
        print(f"  Memory: {result['memory_mb']:.2f} MB")
        print(f"  Throughput: {result['throughput']:.2f} tokens/sec")
    
    # 创建结果表格
    df = pd.DataFrame(results)
    
    # 重排列
    cols = ['name', 'input_shape', 'time_mean', 'time_std', 'memory_mb', 
            'throughput', 'flops']
    df = df[cols]
    
    return df


def main():
    """主函数"""
    print("=" * 80)
    print("RWKV Performance Benchmark")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # 测试配置
    configs = [
        # RWKV 不同输入规模
        {
            'name': 'RWKV (B=1, T=512, D=512)',
            'module': RWKV_TimeMix(dim=512, layer_id=0),
            'input_shape': (1, 512, 512),
        },
        {
            'name': 'RWKV (B=4, T=512, D=512)',
            'module': RWKV_TimeMix(dim=512, layer_id=0),
            'input_shape': (4, 512, 512),
        },
        {
            'name': 'RWKV (B=1, T=2048, D=512)',
            'module': RWKV_TimeMix(dim=512, layer_id=0),
            'input_shape': (1, 2048, 512),
        },
        
        # 对比标准 Attention
        {
            'name': 'NativeAttention (B=1, T=512, D=512)',
            'module': NativeAttention(dim=512, num_heads=8),
            'input_shape': (1, 512, 512),
        },
        {
            'name': 'NativeAttention (B=1, T=1024, D=512)',
            'module': NativeAttention(dim=512, num_heads=8),
            'input_shape': (1, 1024, 512),
        },
    ]
    
    # 运行对比
    df = run_comparison(configs, device=device)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 保存结果
    output_file = 'benchmark_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # 分析结果
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)
    
    # 提取 RWKV 和 Attention 结果
    rwkv_results = df[df['name'].str.contains('RWKV')]
    attn_results = df[df['name'].str.contains('NativeAttention')]
    
    if len(rwkv_results) > 0:
        avg_rwkv_time = rwkv_results['time_mean'].mean()
        print(f"RWKV average time: {avg_rwkv_time:.3f} ms")
    
    if len(attn_results) > 0:
        avg_attn_time = attn_results['time_mean'].mean()
        print(f"Attention average time: {avg_attn_time:.3f} ms")
    
    print("\nKey Observations:")
    print("1. RWKV has O(N) complexity vs Attention's O(N^2)")
    print("2. RWKV memory usage is constant during inference")
    print("3. For long sequences (>1K), RWKV is typically faster")


if __name__ == "__main__":
    main()
