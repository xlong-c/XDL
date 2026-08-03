"""
扩散模型中的JVP(Jacobian-Vector Product)机制详解与演示
JVP (Jacobian-Vector Product) 即雅可比-向量积, 在扩散模型训练和推理中发挥着重要作用。

主要应用包括：
1. 计算噪声预测模型关于输入图像的梯度
2. 计算潜在空间变换的雅可比矩阵
3. 估计数据分布的变化
4. Fisher信息矩阵向量积的高效计算
5. 扩散模型采样轨迹的稳定性分析
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义简单的时间步嵌入层


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


# 定义简单的去噪UNet模型
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=64, out_channels=3, image_height=32, image_width=32, num_res_blocks=2):
        super().__init__()

        self.time_embed = TimeEmbedding(model_channels)
        self.image_height = image_height
        self.image_width = image_width
        self.in_channels = in_channels

        # 计算展平后的图像维度
        flat_image_dim = in_channels * image_height * image_width

        # 简单的线性层作为去噪网络
        self.denoise_net = nn.Sequential(
            nn.Linear(flat_image_dim + model_channels, flat_image_dim),  # 输入：展平图像+时间嵌入, 输出：展平图像
            nn.ReLU(),
            nn.Linear(flat_image_dim, flat_image_dim),
            nn.ReLU(),
            nn.Linear(flat_image_dim, flat_image_dim)  # 输出维度与展平图像相同
        )

    def forward(self, x, t):
        # x: [batch_size, channels, height, width]
        # t: [batch_size]
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # 展平图像

        # 获取时间嵌入
        t_emb = self.time_embed(t)

        # 合并时间和图像特征
        combined_input = torch.cat([x_flat, t_emb], dim=1)

        # 通过去噪网络
        noise_pred = self.denoise_net(combined_input)

        # 重塑回原始图像形状
        noise_pred = noise_pred.view_as(x)

        return noise_pred


def jvp_demo():
    """
    扩散模型中JVP机制的演示
    """

    print("=== 扩散模型中的JVP机制演示 ===\n")

    # 创建模型实例
    model = SimpleUNet()

    # 模拟一个批次的数据
    batch_size = 2
    channels = 3
    height = 32
    width = 32
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)
    t = torch.randint(0, 1000, (batch_size,))

    print(f"输入张量形状: {x.shape}")
    print(f"时间步: {t}\n")

    # 演示JVP的计算
    print("1. 前向传播计算预测噪声")
    pred_noise = model(x, t)
    print(f"预测噪声形状: {pred_noise.shape}\n")

    # 计算JVP：雅可比矩阵与向量的乘积
    print("2. 计算JVP：输入x的梯度")

    # 使用PyTorch的torch.autograd.functional.jvp计算JVP
    # 首先定义目标函数(这里以L2损失为例)
    def loss_fn(x_input):
        pred = model(x_input, t)
        return torch.sum(pred ** 2)  # 简单的L2损失为例

    # 计算损失相对于x的梯度 (这实际上是VJP - Vector-Jacobian Product)
    loss = loss_fn(x)
    grad_x = torch.autograd.grad(loss, x, retain_graph=True)[0]
    print(f"x的梯度形状: {grad_x.shape}")
    print(f"x的梯度统计: mean={grad_x.mean():.6f}, std={grad_x.std():.6f}\n")

    # 构造一个随机向量作为JVP中的"向量"部分
    vector = torch.randn_like(x)

    # 使用JVP计算损失相对于输入的变化率
    # loss_fn聚合了输出, 所以我们期望得到一个标量结果
    _, jvp_result = torch.autograd.functional.jvp(loss_fn, (x,), (vector,))
    assert isinstance(jvp_result, torch.Tensor)
    print(f"JVP结果形状: {jvp_result.shape}")
    if jvp_result.numel() > 1:
        print(f"JVP结果统计: mean={jvp_result.mean():.6f}, std={jvp_result.std():.6f}\n")
    else:
        print(f"JVP结果统计: value={jvp_result.item():.6f}\n")

    print("3. JVP在扩散模型中的实际应用示例")

    # 在扩散模型的采样步骤中, 我们经常需要计算当前状态对先前状态的导数
    # 这种计算可以通过JVP高效实现

    # 模拟扩散过程中的一步
    alpha_bar = 0.8  # 前向扩散过程中的累积方差参数
    noise = torch.randn_like(x)

    # 前向扩散一步: q(x_t | x_0) = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps
    x_noisy = torch.sqrt(torch.tensor(alpha_bar)) * x + \
        torch.sqrt(torch.tensor(1 - alpha_bar)) * noise

    # 模拟去噪一步 (反向扩散过程的一部分)
    _ = model(x_noisy, t)

    # 计算从噪声状态到去噪状态的JVP
    def denoise_step(x_input):
        return model(x_input, t)

    _, jvp_denoise = torch.autograd.functional.jvp(
        denoise_step, (x_noisy,), (vector,))
    assert isinstance(jvp_denoise, torch.Tensor)
    print(f"去噪步骤的JVP形状: {jvp_denoise.shape}")
    print(
        f"去噪步骤的JVP统计: mean={jvp_denoise.mean():.6f}, std={jvp_denoise.std():.6f}\n")

    print("4. JVP与传统梯度计算的比较")

    # 使用传统方式计算完整梯度矩阵(内存密集型)
    print(f"如果计算完整的Jacobian矩阵 (对于形状{x.shape}的输入):")
    elements_count = torch.numel(x)
    print(
        f"  需要存储的参数数量: {elements_count}^2 ≈ {elements_count**2 / 1e6:.2f} 百万个参数")
    print("  对于大图像, 这种方法会非常消耗内存\n")

    # JVP只计算J*v, 避免显式构造完整的Jacobian矩阵
    print(f"而JVP只计算J*v, 输出形状与输入相同({x.shape}), 内存效率更高")

    return model, x, t, grad_x, jvp_result, jvp_denoise


def fisher_information_matrix_vjp_example():
    """
    演示JVP在计算Fisher信息矩阵向量积中的应用
    在扩散模型训练中, Fisher信息矩阵向量积可用于自然梯度下降
    """
    print("\n=== Fisher信息矩阵向量积的JVP计算 ===\n")

    # 创建模型和数据
    model = SimpleUNet()

    batch_size = 1
    x = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
    t = torch.tensor([500])

    # 定义对数似然函数(模拟扩散模型中的负对数似然)
    def log_likelihood(theta_params):
        # 这里简化为一个基于模型输出的函数
        output = model(x, t)
        return -0.5 * torch.sum(output ** 2)

    # 计算log_likelihood相对于模型参数的梯度
    # 这相当于score function (梯度log概率)
    loglik_value = log_likelihood(model.parameters().__next__())  # 仅为演示
    print(f"对数似然值: {loglik_value.item():.4f}")

    # 在实际应用中, 我们会计算 score vector (梯度) 和其JVP
    # 下面演示一个简化的Fisher VJP计算
    def score_function(params_vec):
        # 简化的score function计算
        # 在实际应用中这会是log p(x|theta)相对于theta的梯度
        return torch.sum(params_vec ** 2)  # 简化模拟

    # 构造一个参数向量
    dummy_params = torch.randn(100)  # 模拟参数向量
    vector = torch.randn_like(dummy_params)

    # 计算score function的JVP
    _, fisher_vjp = torch.autograd.functional.jvp(
        score_function, (dummy_params,), (vector,))
    assert isinstance(fisher_vjp, torch.Tensor)

    print(f"Fisher信息矩阵向量积形状: {fisher_vjp.shape}")
    if fisher_vjp.numel() > 1:
        print(f"Fisher VJP统计: mean={fisher_vjp.mean():.6f}, std={fisher_vjp.std():.6f}")
    else:
        print(f"Fisher VJP统计: value={fisher_vjp.item():.6f}")

    print("\n在实际扩散模型中, Fisher VJP可用于:")
    print("- 自然梯度下降, 改善收敛性质")
    print("- 参数空间中的预条件优化")
    print("- 更好的不确定性量化")


def sensitivity_analysis_with_jvp():
    """
    使用JVP进行模型敏感性分析
    这在扩散模型中可用于理解输入扰动对输出的影响
    """
    print("\n=== 使用JVP进行敏感性分析 ===\n")

    model = SimpleUNet()

    batch_size = 1
    x = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
    t = torch.tensor([100])

    def model_output_norm(input_tensor):
        # 计算模型输出的L2范数作为敏感性指标
        output = model(input_tensor, t)
        return torch.norm(output, p=2)

    # 构造扰动向量
    perturbation_direction = torch.randn_like(x)

    # 计算敏感性：输出相对于输入在扰动方向上的变化率
    func_output, sensitivity_jvp = torch.autograd.functional.jvp(
        model_output_norm,
        (x,),
        (perturbation_direction,)
    )
    assert isinstance(sensitivity_jvp, torch.Tensor)

    print(f"敏感性(输出相对于输入扰动的变化率): {sensitivity_jvp.item():.6f}")

    print("\n敏感性分析在扩散模型中的应用:")
    print("- 检测模型对输入扰动的鲁棒性")
    print("- 识别模型的敏感区域")
    print("- 优化采样策略")


def advanced_jvp_example():
    """
    更高级的JVP应用：计算扩散模型中参数的梯度
    """
    print("\n=== 高级JVP应用：参数梯度计算 ===\n")

    # 创建模型
    model = SimpleUNet()

    # 获取模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params}\n")

    # 创建输入
    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([500])
    target = torch.randn_like(x)

    # 定义损失函数
    def compute_loss(params_dict, buffers_dict, x_input, t_input, target_input):
        # 使用torch.func.functional_call来调用模型
        import torch.func as func
        output = func.functional_call(
            model, (params_dict, buffers_dict), x_input, t_input)
        return torch.mean((output - target_input)**2)

    # 分离参数和缓冲区
    buffers_dict = dict(model.named_buffers())

    # 计算相对于参数的JVP
    def param_loss(params_subset):
        # 注意：这里仅简化演示, 实际实现会更复杂
        return compute_loss(params_subset, buffers_dict, x, t, target)

    # 为了演示, 我们创建一个小的参数子集用于JVP计算
    small_param = next(model.parameters())  # 使用第一个参数

    # 计算参数方向上的JVP
    def single_param_fn(param_val):
        old_val = next(model.parameters())
        with torch.no_grad():
            old_val.copy_(param_val)
        loss_val = torch.mean(model(x, t)**2)  # 简化的损失函数
        return loss_val

    jvp_param, = torch.autograd.grad(single_param_fn(small_param),
                                     small_param,
                                     grad_outputs=torch.ones(()))

    print(f"参数方向JVP: 均值={jvp_param.mean():.6f}, 标准差={jvp_param.std():.6f}")
    print(f"参数形状: {small_param.shape}")


def jvp_for_sampling():
    """
    JVP在扩散模型采样中的具体应用
    """
    print("\n=== JVP在扩散模型采样中的应用 ===\n")

    # 初始化模型和输入
    model = SimpleUNet()

    batch_size = 1
    channels = 3
    height = 32
    width = 32

    # 从纯噪声开始采样
    x_T = torch.randn(batch_size, channels, height, width)

    # 定义采样中的某一步(例如第500步到499步)
    t = torch.tensor([499])  # 当前时间步

    # 在扩散过程的单步中, 我们需要计算：
    # p(x_{t-1} | x_t) 的均值和方差
    # 这涉及到复杂的梯度计算, 其中JVP可以提高效率

    # 演示在采样过程中如何利用JVP
    def sample_step_fn(x_t, time_step):
        """单步采样函数"""
        # 预测噪声
        pred_eps = model(x_t, time_step)
        return pred_eps

    # 构建要计算JVP的函数
    def sampling_gradient(x_input):
        return sample_step_fn(x_input, t)

    # 计算JVP - 这模拟了在采样过程中评估x_t对最终输出影响的方式
    vector = torch.randn_like(x_T)

    # 计算JVP: Jacobian of sampling_gradient w.r.t. x_input multiplied by vector
    _, jvp_sample = torch.autograd.functional.jvp(
        lambda x: sampling_gradient(x).sum(),  # 将输出聚合以便求导
        (x_T,),
        (vector,)
    )
    assert isinstance(jvp_sample, torch.Tensor)

    print(f"采样步骤的JVP形状: {jvp_sample.shape}")
    if jvp_sample.numel() > 1:
        print(f"采样步骤的JVP统计: mean={jvp_sample.mean():.6f}, std={jvp_sample.std():.6f}")
    else:
        print(f"采样步骤的JVP统计: value={jvp_sample.item():.6f}")

    print("\n在实际的扩散模型采样中, JVP可用于:")
    print("- 高效计算Fisher信息矩阵向量积, 加速参数优化")
    print("- 提供对采样轨迹稳定性的洞察")
    print("- 帮助分析模型的敏感性和鲁棒性")


def visualize_diffusion_process():
    """
    可视化扩散过程中的JVP效果
    """
    print("\n=== 可视化扩散过程 ===\n")

    # 简化的扩散过程模拟
    T = 1000  # 总时间步
    betas = torch.linspace(0.0001, 0.02, T)  # beta值从0.0001到0.02
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    # 选择几个关键时间点
    time_steps = [0, 100, 250, 500, 750, 999]

    print("扩散过程的关键时间步及对应的噪声水平:")
    for t in time_steps:
        print(
            f"时间步 {t}: alpha_bar = {alphas_bar[t]:.4f}, sqrt(1-alpha_bar) = {torch.sqrt(1-alphas_bar[t]):.4f}")

    # 绘制alpha_bar随时间变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(alphas_bar.numpy(), label='alpha_bar_t')
    plt.xlabel('Time Step t')
    plt.ylabel('Cumulative Variance alpha_bar_t')
    plt.title('Cumulative Variance Change in Diffusion Process')
    plt.grid(True)
    plt.legend()
    plt.savefig('diffusion_alphabar.png')
    print("\n已生成扩散过程的可视化图: diffusion_alphabar.png")


if __name__ == "__main__":
    # 运行基本的JVP演示
    model, x, t, grad_x, jvp_result, jvp_denoise = jvp_demo()

    # 运行高级JVP示例
    advanced_jvp_example()

    # 运行JVP在采样中的应用示例
    jvp_for_sampling()

    # 运行Fisher信息矩阵VJP示例
    fisher_information_matrix_vjp_example()

    # 运行敏感性分析示例
    sensitivity_analysis_with_jvp()

    # 运行扩散过程可视化
    visualize_diffusion_process()

    print("\n=== JVP在扩散模型中的重要性总结 ===")
    print("1. 内存效率: JVP避免了显式计算完整的Jacobian矩阵, 节省大量内存")
    print("2. 计算效率: 对于高维输入, JVP的计算复杂度远低于完整梯度矩阵")
    print("3. 优化应用: 在参数更新中, JVP可以高效计算Fisher信息矩阵向量积")
    print("4. 采样质量: 通过准确的梯度信息, 提高扩散模型采样的质量")
    print("5. 敏感性分析: 帮助理解模型对输入扰动的响应")
    print("6. 稳定性分析: 评估采样算法的数值稳定性")
