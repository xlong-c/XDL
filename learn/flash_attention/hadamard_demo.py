# FlashAttention v4: Hadamard 变换与精度保持演示
# 核心：对比 Hadamard 变换如何将离散值“平铺”并改善 FP8 量化精度。

import torch


def simulate_outlier_data(dim=1024):
    """
    模拟激活值数据，通常会有几个极大的离群值 (Outliers)。
    """
    # 基础正态分布数据
    x = torch.randn(dim)
    # 人为插入离群值
    x[0] = 50.0  # 极大的离群值，会导致量化溢出
    x[1] = -45.0
    return x


def quantize_to_fp8(x):
    """
    模拟简单的 FP8 量化：将数据缩放到 [-127, 127] 的整数范围。
    离群值会导致缩放因子变大，进而导致小数值精度完全丢失。
    """
    scale = 127.0 / x.abs().max()
    return (x * scale).round().clamp(-127, 127)


def hadamard_transform_vector(x):
    """递归实现的 Hadamard 变换"""
    n = x.shape[0]
    if n == 1:
        return x
    left = hadamard_transform_vector(x[: n // 2])
    right = hadamard_transform_vector(x[n // 2 :])
    return torch.cat([left + right, left - right]) / (2**0.5)


# --- 1. 场景对比 ---
dim = 1024
x = simulate_outlier_data(dim)

# 情况 A: 直接量化误差
quant_raw = quantize_to_fp8(x)
loss_raw = torch.mean((quant_raw.float() / (127.0 / x.abs().max()) - x) ** 2)

# 情况 B: Hadamard 变换后量化并还原对比
# 1. 变换到 Hadamard 空间
x_h = hadamard_transform_vector(x)
# 2. 计算该空间的缩放因子
scale_h = 127.0 / x_h.abs().max()
# 3. 量化
quant_h = quantize_to_fp8(x_h)
# 4. 还原到原始空间 (先反量化，再逆变换)
# Hadamard 矩阵是正交的，逆变换即自身除以缩放因子
x_recovered = hadamard_transform_vector(quant_h.float() / scale_h)

loss_h = torch.mean((x_recovered - x) ** 2)

print(f"原始数据最大值: {x.abs().max():.2f}")
print(f"Hadamard 变换后最大值: {x_h.abs().max():.2f}")
print(f"直接量化 MSE 损失: {loss_raw:.4f}")
print(f"Hadamard 量化还原后 MSE 损失: {loss_h:.4f}")

"""
结论：
1. 离群值被分散后，数据分布的“峰值”显著下降，量化缩放因子更合理。
2. 虽然引入了 Hadamard 变换增加了计算量 (O(d log d))，但量化误差显著减小，
   使得 FP8 矩阵乘法能够保持接近 BF16 的精度。
"""
