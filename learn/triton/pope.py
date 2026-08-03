import torch
import torch.nn as nn
import math

class PoPETorch(nn.Module):
    """
    修复版PyTorch-PoPE：解决可学习偏置δ范围不符合预期的问题
    关键优化：支持Uniform(-2π, 0)初始化，确保约束逻辑生效
    """
    def __init__(
        self, 
        d_model: int, 
        theta_base: float = 10000.0, 
        use_learnable_delta: bool = True,
        delta_init_type: str = "uniform"  # 新增：delta初始化方式（"zero"或"uniform"）
    ):
        super().__init__()
        self.d_model = d_model
        self.theta_base = theta_base
        self.use_learnable_delta = use_learnable_delta
        self.delta_init_type = delta_init_type  # 记录初始化方式
        
        # 1. 预计算频率θ_c（论文公式4）
        c = torch.arange(1, d_model + 1, dtype=torch.float32)
        self.theta = theta_base ** ((c - 1) / d_model)
        self.theta = nn.Parameter(self.theta, requires_grad=False)
        
        # 2. 初始化可学习偏置δ（论文3.1节）
        if use_learnable_delta:
            if delta_init_type == "zero":
                self.delta = nn.Parameter(torch.zeros(d_model))  # 零初始化（利于长度外推）
            elif delta_init_type == "uniform":
                # 均匀分布初始化（-2π ~ 0），更易验证约束效果
                self.delta = nn.Parameter(torch.rand(d_model) * (-2 * math.pi))
            else:
                raise ValueError("delta_init_type must be 'zero' or 'uniform'")
            
            # 约束δ在[-2π, 0]范围内（核心修复：确保Hardtanh参数正确）
            self.delta_constraint = nn.Hardtanh(min_val=-2 * math.pi, max_val=0.0)
        else:
            self.delta = None
        
        # 3. Softplus激活（论文公式3）
        self.softplus = nn.Softplus()

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, _, _ = q.shape
        
        # 生成位置序列
        positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        positions = positions.view(1, 1, seq_len, 1)
        
        # 计算幅度（μ）和相位（φ）
        mu_q = self.softplus(q)
        mu_k = self.softplus(k)
        
        phi_q = positions * self.theta.view(1, 1, 1, self.d_model)
        phi_k = positions * self.theta.view(1, 1, 1, self.d_model)
        
        # 应用可学习偏置δ（约束生效）
        if self.use_learnable_delta:
            delta = self.delta_constraint(self.delta)  # 强制裁剪到[-2π, 0]
            phi_k = phi_k + delta.view(1, 1, 1, self.d_model)
        
        # 转换为复数值（实部+虚部）
        x_q = mu_q * torch.cos(phi_q)
        y_q = mu_q * torch.sin(phi_q)
        x_k = mu_k * torch.cos(phi_k)
        y_k = mu_k * torch.sin(phi_k)
        
        q_complex = torch.stack([x_q, y_q], dim=-1)
        k_complex = torch.stack([x_k, y_k], dim=-1)
        
        return q_complex, k_complex

    @staticmethod
    def compute_attention_score(q_complex: torch.Tensor, k_complex: torch.Tensor) -> torch.Tensor:
        x_q, y_q = q_complex[..., 0], q_complex[..., 1]
        x_k, y_k = k_complex[..., 0], k_complex[..., 1]
        attn_score = torch.einsum("bhqd, bhkd -> bhqk", x_q, x_k) + torch.einsum("bhqd, bhkd -> bhqk", y_q, y_k)
        return attn_score


# ----------------------
# 测试代码：验证修复效果
# ----------------------
if __name__ == "__main__":
    # 配置参数
    batch_size = 2
    num_heads = 8
    seq_len = 1024
    d_model = 512
    
    # 随机生成q和k
    q = torch.randn(batch_size, num_heads, seq_len, d_model)
    k = torch.randn(batch_size, num_heads, seq_len, d_model)
    
    # 初始化PoPE（使用uniform初始化δ，验证约束效果）
    pope = PoPETorch(
        d_model=d_model, 
        use_learnable_delta=True,
        delta_init_type="uniform"  # 关键：用均匀分布初始化δ
    )
    
    # 前向传播
    q_complex, k_complex = pope(q, k, seq_len=seq_len)
    attn_score = PoPETorch.compute_attention_score(q_complex, k_complex)
    
    # 验证输出维度
    print(f"q_complex形状: {q_complex.shape} → 预期：(2, 8, 1024, 512, 2)")
    print(f"k_complex形状: {k_complex.shape} → 预期：(2, 8, 1024, 512, 2)")
    print(f"注意力分数形状: {attn_score.shape} → 预期：(2, 8, 1024, 1024)")
    
    # 验证可学习偏置δ的范围（修复后符合预期）
    if pope.use_learnable_delta:
        delta = pope.delta_constraint(pope.delta)
        print(f"可学习偏置δ范围: [{delta.min().item():.4f}, {delta.max().item():.4f}] → 预期：[-6.2832, 0.0]")
        
        # 额外验证：手动设置δ超出范围，检查约束是否裁剪
        pope.delta.data = torch.tensor([3.0, -7.0, 0.0], device=pope.delta.device).repeat(d_model//3 + 1)[:d_model]
        delta_clipped = pope.delta_constraint(pope.delta)
        print(f"超出范围的δ裁剪后: [{delta_clipped.min().item():.4f}, {delta_clipped.max().item():.4f}] → 验证约束生效")