# setup.py — 编译 nunchaku GEMM + Attention PyTorch 扩展
# 用法: python setup.py install

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 当前目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 需要编译的源文件
sources = [
    "nunchaku_bridge.cu",
    "bindings.cpp",
]

# CUDA 编译选项
# - SM80+ (Ampere) 是 W4A4/W8A8 kernel 的最低要求
# - 如果需要 SM75 (Turing)，会在 kernel 内自动 fallback 为 m8n8k8 拆分模式
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_80,code=sm_80",  # Ampere (A100)
    "-gencode", "arch=compute_86,code=sm_86",  # Ampere (RTX 3090)
    "-gencode", "arch=compute_89,code=sm_89",  # Ada (RTX 4090)
    "-gencode", "arch=compute_90a,code=sm_90a", # Hopper (H100)
    "--ptxas-options=-v",
    "--use_fast_math",
    "-diag-suppress", "177",  # unused variable warning
]

# 如果 CUDA 版本 >= 12.5，启用 MMA "C" constraint
import subprocess
try:
    nvcc_ver = subprocess.check_output(["nvcc", "--version"]).decode()
    if "V12.5" in nvcc_ver or "V12.6" in nvcc_ver or "V12.7" in nvcc_ver or "V12.8" in nvcc_ver:
        nvcc_flags.append("-DCUDA_125_PLUS")
except:
    pass

setup(
    name="nunchaku_gemm",
    version="0.1.0",
    description="W4A4/W8A8 GEMM and Flash Attention from nunchaku (learning edition)",
    ext_modules=[
        CUDAExtension(
            name="nunchaku_gemm",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
            include_dirs=[BASE_DIR],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    python_requires=">=3.8",
)
