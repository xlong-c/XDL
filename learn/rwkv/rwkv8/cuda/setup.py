"""
Setup script for RWKV CUDA extensions
"""

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# 检查 CUDA 是否可用
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Building CPU-only version.")

# 定义扩展模块
ext_modules = [
    CUDAExtension(
        name='rwkv_cuda',
        sources=[
            'wkv_cuda_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '-arch=sm_70',  # 根据你的 GPU 架构调整
                '-gencode', 'arch=compute_70,code=sm_70',
                '-gencode', 'arch=compute_75,code=sm_75',
                '-gencode', 'arch=compute_80,code=sm_80',
                '-std=c++17',
                '--use_fast_math',
                '-Xcompiler', '-fPIC',
            ],
        },
        extra_link_args=['-s'],  # Strip symbols for smaller binary
    ),
]

setup(
    name='rwkv_cuda',
    version='1.0.0',
    author='RWKV Learner',
    description='CUDA kernels for RWKV WKV computation',
    long_description=open('../README.md').read(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ],
    zip_safe=False,
)
