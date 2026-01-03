import torch
import triton
import triton.language as tl

# -------------------------- 非融合版本（PyTorch原生分步） --------------------------


def non_fused_mul_add_relu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    temp = a * b          # 第一步：乘
    temp2 = temp + c      # 第二步：加
    out = torch.relu(temp2)  # 第三步：ReLU
    return out

# -------------------------- 融合版本（Triton Kernel） --------------------------


@triton.jit
def fused_mul_add_relu_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr  # 编译期常量，控制块大小
):
    # 1. 计算当前线程块处理的索引范围
    pid = tl.program_id(axis=0)  # 程序ID（对应CUDA的blockIdx）
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 线程内索引偏移

    # 2. 加载a/b/c到寄存器（自动优化内存访问）
    # 边界检查：避免越界访问
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)

    # 3. 融合计算：乘 → 加 → ReLU（无中间存储）
    val = a * b + c
    out = tl.where(val > 0, val, 0.0)  # Triton版ReLU

    # 4. 存储结果到全局内存
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_mul_add_relu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # 输入校验
    assert a.is_cuda and b.is_cuda and c.is_cuda, "输入必须是CUDA张量"
    assert a.shape == b.shape == c.shape, "输入形状必须一致"
    n_elements = a.numel()

    # 输出张量初始化
    out = torch.empty_like(a)

    # 配置block大小（Triton自动选最优值，这里手动指定256）
    BLOCK_SIZE = 256
    def grid(meta): return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 启动Triton Kernel
    fused_mul_add_relu_kernel[grid](
        a, b, c, out,
        n_elements,
        BLOCK_SIZE=tl.constexpr(BLOCK_SIZE)
    )
    return out


# -------------------------- 验证与性能测试 --------------------------
if __name__ == "__main__":
    # 1. 初始化数据（CUDA张量）
    device = torch.device("cuda")
    # 增大 N 以减少启动开销的影响 (从 1<<20 增加到 1<<24)
    n = 1 << 24
    a = (torch.randint(-5, 5, (n,), dtype=torch.float32, device=device) + 0.5)
    b = torch.full((n,), 2.0, dtype=torch.float32, device=device)
    c = torch.full((n,), 1.0, dtype=torch.float32, device=device)

    # 2. 计算结果
    out_non_fused = non_fused_mul_add_relu(a, b, c)
    out_fused = fused_mul_add_relu(a, b, c)

    # 3. 验证结果一致性
    is_correct = torch.allclose(out_non_fused, out_fused, atol=1e-5)
    print("结果验证：", "成功" if is_correct else "失败")

    # 4. 性能对比（使用 triton.testing.do_bench 获得稳定结果）
    # do_bench 会自动进行 Warmup，并返回中位数耗时（ms）
    t_non_fused = triton.testing.do_bench(
        lambda: non_fused_mul_add_relu(a, b, c))
    t_fused = triton.testing.do_bench(lambda: fused_mul_add_relu(a, b, c))

    assert isinstance(t_non_fused, float)
    assert isinstance(t_fused, float)
    print(f"数据规模: {n} 元素")
    print(f"非融合版本中位数耗时：{t_non_fused:.4f} ms")
    print(f"融合版本中位数耗时：{t_fused:.4f} ms")
    print(f"性能提升：{t_non_fused/t_fused:.2f} 倍")
