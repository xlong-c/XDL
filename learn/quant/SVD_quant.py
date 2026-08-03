import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import hadamard


def simulate_quantization(matrix, bits, group_size=64):
    """模拟分组量化 (Group-wise Quantization)"""
    if bits >= 16:
        return matrix.astype(np.float16).astype(np.float64)
    orig_shape = matrix.shape
    flat = matrix.flatten()
    n_elements = flat.size
    pad_size = (group_size - (n_elements % group_size)) % group_size
    padded = np.concatenate([flat, np.zeros(pad_size)])
    reshaped = padded.reshape(-1, group_size)
    max_vals = np.max(np.abs(reshaped), axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    q_max = (2**(bits - 1)) - 1
    scales = max_vals / q_max
    quantized = np.round(reshaped / scales).clip(-q_max, q_max)
    dequantized = quantized * scales
    return dequantized.flatten()[:n_elements].reshape(orig_shape)


def get_hadamard_matrix(n):
    h_size = 1
    while h_size < n:
        h_size *= 2
    H = hadamard(h_size).astype(np.float64)
    return H[:n, :n] / np.sqrt(h_size)


class SparseSVDQuant:
    """方案一：四分量压缩 (Sparse + Low-Rank + Residual)"""

    def __init__(self, rank=32, threshold=10.0, bits=4):
        self.rank = rank
        self.threshold = threshold
        self.bits = bits

    def compress(self, matrix):
        # 1. 提取稀疏分量
        mask = np.abs(matrix) > self.threshold
        self.outliers = np.zeros_like(matrix)
        self.outliers[mask] = matrix[mask]

        # 2. 对密集部分做 SVD
        dense = matrix - self.outliers
        svd = TruncatedSVD(n_components=self.rank, random_state=42)
        self.W = svd.fit_transform(dense)
        self.H = svd.components_

        # 3. 计算并量化残差
        res = dense - (self.W @ self.H)
        self.res_q = simulate_quantization(res, self.bits)

    def decompress(self):
        return (self.W @ self.H) + self.res_q + self.outliers


class AbsorbSVDQuant:
    """方案二: SVDQuant (Absorb outliers into Low-Rank + Residual)"""

    def __init__(self, rank=32, bits=4):
        self.rank = rank
        self.bits = bits

    def compress(self, matrix):
        # 1. 直接对带极值的矩阵做 SVD
        svd = TruncatedSVD(n_components=self.rank, random_state=42)
        self.W = svd.fit_transform(matrix)
        self.H = svd.components_

        # 2. 计算并量化残差
        res = matrix - (self.W @ self.H)
        self.res_q = simulate_quantization(res, self.bits)

    def decompress(self):
        return (self.W @ self.H) + self.res_q


class RotationSVDQuant:
    """方案三：旋转压缩 (Hadamard Rotation + Low-Rank + Residual)"""

    def __init__(self, rank=32, bits=4):
        self.rank = rank
        self.bits = bits

    def compress(self, matrix):
        self.dim = matrix.shape[0]
        self.Q = get_hadamard_matrix(self.dim)

        # 1. 旋转矩阵
        rotated = self.Q.T @ matrix

        # 2. 对旋转后的矩阵做 SVD
        svd = TruncatedSVD(n_components=self.rank, random_state=42)
        self.W_r = svd.fit_transform(rotated)
        self.H_r = svd.components_

        # 3. 计算并量化残差
        res_r = rotated - (self.W_r @ self.H_r)
        self.res_r_q = simulate_quantization(res_r, self.bits)

    def decompress(self):
        return self.Q @ ((self.W_r @ self.H_r) + self.res_r_q)


class OptimizedRotationSVDQuant:
    """方案四：优化旋转 (Sparse + Randomized Rotation)"""

    def __init__(self, rank=32, threshold=10.0, bits=4):
        self.rank = rank
        self.threshold = threshold
        self.bits = bits

    def compress(self, matrix):
        self.dim = matrix.shape[0]
        # 1. 提取离群值 (Sparse Component)
        mask = np.abs(matrix) > self.threshold
        self.outliers = np.zeros_like(matrix)
        self.outliers[mask] = matrix[mask]

        dense = matrix - self.outliers

        # 2. 生成随机哈达玛变换 (H @ D)
        # D 是随机的 +/- 1
        self.d_signs = np.random.choice([-1.0, 1.0], size=self.dim)
        self.Q = get_hadamard_matrix(self.dim)

        # 旋转时施加随机符号: Q.T @ (D * dense)
        rotated = self.Q.T @ (self.d_signs[:, None] * dense)

        # 3. 对旋转后的平滑矩阵做 SVD
        svd = TruncatedSVD(n_components=self.rank, random_state=42)
        self.W_r = svd.fit_transform(rotated)
        self.H_r = svd.components_

        # 4. 量化残差
        res_r = rotated - (self.W_r @ self.H_r)
        self.res_r_q = simulate_quantization(res_r, self.bits)

    def decompress(self):
        # 逆旋转: D * (Q @ (LR + Res))
        recon_rotated = (self.W_r @ self.H_r) + self.res_r_q
        dense_recon = self.d_signs[:, None] * (self.Q @ recon_rotated)
        return dense_recon + self.outliers

# --- 测试代码 ---


if __name__ == "__main__":
    np.random.seed(42)
    dim = 1024
    rank = 16
    bits = 4
    # 1. 生成测试权重 B (我们要压缩的对象)
    B_orig = np.random.randn(dim, dim).astype(np.float64)
    for _ in range(100):
        i, j = np.random.randint(0, dim), np.random.randint(0, dim)
        B_orig[i, j] += np.random.choice([-80.0, 80.0])
    # 2. 生成输入激活 A (用于测试矩阵乘法误差)
    # A 通常是较小的正态分布
    A_input = np.random.randn(500, dim).astype(np.float64)
    C_ref = A_input @ B_orig
    print(
        f"压缩测试 | 权重: {dim}x{dim}, 输入: 500x{dim} | {bits}-bit | Rank: {rank}\n")
    methods = [
        ("Sparse+LR+Res (4-分量)", SparseSVDQuant(rank=rank, threshold=10.0, bits=bits)),
        ("SVDQuant (低秩吸收)", AbsorbSVDQuant(rank=rank, bits=bits)),
        ("Rotation+SVD (旋转方案)", RotationSVDQuant(rank=rank, bits=bits)),
        ("Optimized Rotation (优化旋转)", OptimizedRotationSVDQuant(
            rank=rank, threshold=10.0, bits=bits))
    ]

    for name, model in methods:
        # 压缩权重
        model.compress(B_orig)
        B_recon = model.decompress()
        # 1. 权重重构误差 (Weight Error)
        w_rel_err = np.linalg.norm(B_orig - B_recon) / np.linalg.norm(B_orig)
        w_snr = 20 * np.log10(np.linalg.norm(B_orig) /
                              np.linalg.norm(B_orig - B_recon))
        # 2. 矩阵乘法结果误差 (Output Error: C = A @ B)
        C_recon = A_input @ B_recon
        c_rel_err = np.linalg.norm(C_ref - C_recon) / np.linalg.norm(C_ref)
        c_snr = 20 * np.log10(np.linalg.norm(C_ref) /
                              np.linalg.norm(C_ref - C_recon))
        print(f"{name}:")
        print(f"  [权重重构] SNR: {w_snr:.2f} dB | 相对误差: {w_rel_err:.6f}")
        print(f"  [乘法输出] SNR: {c_snr:.2f} dB | 相对误差: {c_rel_err:.6f}")
        print("-" * 50)
