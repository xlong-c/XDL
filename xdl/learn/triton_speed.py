import torch
import triton
import triton.language as tl

# 检查硬件环境


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(device)
    print(f"测试设备: {name}")
    print(f"计算能力: {capability[0]}.{capability[1]}")
    return capability

# 通用 Triton Matmul 内核


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256,
                      'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :]
                    < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None]
                    < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 处理输出精度
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, out_dtype=torch.float16, acc_dtype=torch.float32):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    def grid(META): return (triton.cdiv(
        M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # 映射 Torch dtype 到 Triton dtype
    if acc_dtype == torch.float32:
        acc_type = tl.float32
    elif acc_dtype == torch.float16:
        acc_type = tl.float16
    elif acc_dtype == torch.bfloat16:
        acc_type = tl.bfloat16
    else:
        acc_type = tl.float32

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACC_TYPE=acc_type,
    )
    return c


def benchmark():
    capability = check_gpu()
    # 4070 Ti Super 建议测试规模
    SIZE = 8192
    M, N, K = SIZE, SIZE, SIZE

    test_types = [
        ("FP32", torch.float32, torch.float32, torch.float32),
        ("TF32", torch.float32, torch.float32, torch.float32),
        ("FP16 (Acc32)", torch.float16, torch.float16, torch.float32),
        ("FP16 (Acc16)", torch.float16, torch.float16, torch.float16),
        ("BF16 (Acc32)", torch.bfloat16, torch.bfloat16, torch.float32),
        ("BF16 (Acc16)", torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ]

    # FP8 检测
    if capability[0] >= 9 or (capability[0] == 8 and capability[1] >= 9):
        try:
            test_types.append(
                ("FP8 (Acc32)", torch.float8_e4m3fn, torch.float16, torch.float32))
            test_types.append(
                ("FP8 (Acc16)", torch.float8_e4m3fn, torch.float16, torch.float16))
        except AttributeError:
            pass

    print(f"\n{'Precision':<15} | {'Latency (ms)':>15} | {'TFLOPS':>12}")
    print("-" * 48)

    for name, in_dtype, out_dtype, acc_dtype in test_types:
        # 特殊处理: NVIDIA GPU 上的 Acc16
        is_nvidia = torch.cuda.get_device_name().lower().find("nvidia") != -1
        is_acc16 = (acc_dtype == torch.float16 or acc_dtype == torch.bfloat16)
        
        if is_nvidia and is_acc16:
            print(f"{name:<15} | {'Not Supported':>15} | NVIDIA requires Acc32 for tl.dot")
            continue

        if name == "TF32":
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False

        try:
            # 准备数据
            # 对于 FP8, 输入需要特殊处理
            if "FP8" in name:
                # 随机生成 FP16 然后转换到 FP8
                a = torch.randn((M, K), device='cuda', dtype=torch.float16).to(in_dtype)
                # Triton matmul 要求 b 是转置后的连续内存, 或者在加载时处理
                # 这里简单处理
                b = torch.randn((K, N), device='cuda', dtype=torch.float16).to(in_dtype)
            else:
                a = torch.randn((M, K), device='cuda', dtype=in_dtype)
                b = torch.randn((K, N), device='cuda', dtype=in_dtype)

            # 预热
            for _ in range(5):
                matmul(a, b, out_dtype=out_dtype, acc_dtype=acc_dtype)

            # 计时
            latency = triton.testing.do_bench(
                lambda: matmul(a, b, out_dtype=out_dtype, acc_dtype=acc_dtype))

            # 确保 latency 不为 None
            if latency is None:
                raise ValueError("Benchmark returned None")

            if isinstance(latency, (list, tuple)):
                ms = float(latency[0])
            else:
                ms = float(latency)

            tflops = (2.0 * M * N * K) / (ms * 1e-3 * 1e12)
            print(f"{name:<15} | {ms:15.4f} | {tflops:12.2f}")

        except Exception as e:
            # 打印简短错误
            error_msg = str(e).split('\n')[0]
            print(f"{name:<15} | {'Error':>15} | {error_msg[:30]}")

    print("-" * 48)
    print("备注: 1. 4070 Ti Super 在 FP8 模式下理论峰值可达约 140+ TFLOPS。")
    print("      2. NVIDIA GPU 硬件上, tl.dot 强制要求使用 FP32 累加以利用 Tensor Core。")
    print("      3. FP16 (Acc16) 在 NVIDIA 后端不被 Triton 支持, 故跳过。")


if __name__ == "__main__":
    benchmark()
