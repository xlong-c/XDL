"""
RWKV CUDA 模块

提供高性能的 WKV 计算 CUDA 实现
"""

from .wkv_interface import (
    WKVFunction,
    WKVLayer,
    wkv_forward_parallel_torch,
    CUDA_AVAILABLE
)

__all__ = [
    'WKVFunction',
    'WKVLayer', 
    'wkv_forward_parallel_torch',
    'CUDA_AVAILABLE',
]
