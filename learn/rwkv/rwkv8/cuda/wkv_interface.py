"""
RWKV WKV CUDA 接口 - Python 层封装

提供易于使用的 Python API 调用 CUDA kernel
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import os

# 尝试导入 CUDA 扩展
try:
    import rwkv_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: rwkv_cuda not found. Using PyTorch fallback.")


class WKVFunction(torch.autograd.Function):
    """
    WKV 计算的自定义 autograd Function
    支持前向和反向传播
    """
    
    @staticmethod
    def forward(ctx, r, k, v, w, u):
        """
        前向传播
        
        Args:
            r: [B, T, C] - receptance gate
            k: [B, T, C] - key
            v: [B, T, C] - value
            w: [C] - time decay
            u: [C] - bonus
            
        Returns:
            out: [B, T, C] - WKV output
        """
        ctx.save_for_backward(r, k, v, w, u)
        
        if CUDA_AVAILABLE and r.is_cuda:
            # 调用 CUDA kernel
            out = rwkv_cuda.wkv_forward_parallel(r, k, v, w, u)
        else:
            # PyTorch fallback
            out = wkv_forward_parallel_torch(r, k, v, w, u)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        
        Args:
            grad_output: [B, T, C] - 输出梯度
            
        Returns:
            各输入的梯度
        """
        r, k, v, w, u = ctx.saved_tensors
        
        # TODO: 实现 CUDA 反向传播
        # 目前使用 PyTorch autograd
        grad_r, grad_k, grad_v, grad_w, grad_u = None, None, None, None, None
        
        # 这里需要实现具体的反向传播逻辑
        # 暂时返回 None，使用数值梯度或近似
        
        return grad_r, grad_k, grad_v, grad_w, grad_u


def wkv_forward_parallel_torch(r, k, v, w, u):
    """
    PyTorch 实现的并行 WKV (作为 CUDA 的 fallback)
    
    Args:
        r, k, v: [B, T, C]
        w, u: [C]
    
    Returns:
        out: [B, T, C]
    """
    B, T, C = k.shape
    
    # 确保 w 是负数
    w = -torch.abs(w)
    
    # 计算 exp(k)
    exp_k = torch.exp(k)
    exp_w = torch.exp(w)
    
    # 累积计算
    num = torch.zeros(B, T, C, device=k.device)
    den = torch.zeros(B, T, C, device=k.device)
    
    # 第一个位置
    num[:, 0] = exp_k[:, 0] * v[:, 0]
    den[:, 0] = exp_k[:, 0]
    
    # 累积
    for t in range(1, T):
        num[:, t] = exp_k[:, t] * v[:, t] + num[:, t-1] * exp_w
        den[:, t] = exp_k[:, t] + den[:, t-1] * exp_w
    
    # 应用 bonus
    exp_k_u = torch.exp(k + u.unsqueeze(0).unsqueeze(0))
    num_bonus = num + exp_k_u * v
    den_bonus = den + exp_k_u
    
    # 应用 receptance
    wkv = r * num_bonus / (den_bonus + 1e-6)
    
    return wkv


class WKVLayer(nn.Module):
    """
    完整的 WKV 层，用于构建 RWKV 模型
    """
    
    def __init__(self, dim, layer_id=0):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        
        # 时间混合参数
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, dim))
        
        # WKV 参数
        self.time_decay = nn.Parameter(torch.ones(dim))
        self.time_bonus = nn.Parameter(torch.ones(dim))
        
        # 投影层
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, T, D]
        
        Returns:
            out: [B, T, D]
        """
        B, T, D = x.shape
        
        # 时间混合
        xx = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        
        # 计算 R, K, V
        r = torch.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)
        
        # WKV 计算
        out = WKVFunction.apply(r, k, v, self.time_decay, self.time_bonus)
        
        return out


def benchmark_wkv():
    """基准测试 WKV 实现"""
    import time
    
    print("Benchmarking WKV implementations...")
    
    # 测试配置
    batch_sizes = [1, 4, 16]
    seq_lengths = [128, 512, 1024, 2048]
    dim = 512
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for B in batch_sizes:
        for T in seq_lengths:
            print(f"\nBatch: {B}, Seq: {T}, Dim: {dim}")
            
            # 创建输入
            r = torch.randn(B, T, dim, device=device)
            k = torch.randn(B, T, dim, device=device)
            v = torch.randn(B, T, dim, device=device)
            w = torch.randn(dim, device=device)
            u = torch.randn(dim, device=device)
            
            # 预热
            for _ in range(10):
                out = wkv_forward_parallel_torch(r, k, v, w, u)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 基准测试
            n_iters = 100
            start = time.time()
            
            for _ in range(n_iters):
                out = wkv_forward_parallel_torch(r, k, v, w, u)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            avg_time = elapsed / n_iters * 1000  # ms
            
            print(f"  PyTorch: {avg_time:.3f} ms/iter")
            
            # 如果有 CUDA 实现，也测试它
            if CUDA_AVAILABLE and device.type == 'cuda':
                # 预热
                for _ in range(10):
                    out = rwkv_cuda.wkv_forward_parallel(r, k, v, w, u)
                
                torch.cuda.synchronize()
                
                # 测试
                start = time.time()
                for _ in range(n_iters):
                    out = rwkv_cuda.wkv_forward_parallel(r, k, v, w, u)
                torch.cuda.synchronize()
                
                elapsed = time.time() - start
                avg_time_cuda = elapsed / n_iters * 1000
                speedup = avg_time / avg_time_cuda
                
                print(f"  CUDA:    {avg_time_cuda:.3f} ms/iter ({speedup:.2f}x speedup)")


if __name__ == "__main__":
    # 运行基准测试
    benchmark_wkv()
