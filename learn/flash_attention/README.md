# Softmax to FlashAttention-2

这个目录是一条教学路线, 目标是从最普通的 `softmax` 一直推到
FlashAttention-2 的 forward kernel 思路。每一节都有一份讲义 `.md` 和一份
对应代码 `.py`。所有主实现都写成 Triton kernel; PyTorch 只用于创建 tensor
和做 correctness reference。

这不是生产级替代品。这里的 Triton kernel 只覆盖教学所需的 forward 路径,
没有 backward, dropout, ALiBi, varlen, autotune, FP8 等工程细节。

## 学习顺序

1. [01_softmax.md](01_softmax.md) /
   [01_softmax.py](01_softmax.py)
   - 从 naive softmax 到 safe softmax, 理解为什么要减去 row max。
2. [02_online_softmax.md](02_online_softmax.md) /
   [02_online_softmax.py](02_online_softmax.py)
   - 把 softmax 的分母改成 streaming 更新, 这是 FlashAttention 的核心小积木。
3. [03_attention_baseline.md](03_attention_baseline.md) /
   [03_attention_baseline.py](03_attention_baseline.py)
   - 用 Triton 显式 materialize `N x N` logits/probs, 看清楚为什么吃显存。
4. [04_flash_attention_v1.md](04_flash_attention_v1.md) /
   [04_flash_attention_v1.py](04_flash_attention_v1.py)
   - 用 Triton block loop 写一个精确的 FlashAttention v1 教学版。
5. [05_triton_softmax.md](05_triton_softmax.md) /
   [05_triton_softmax.py](05_triton_softmax.py)
   - 用 Triton 写 fused row softmax, 把一行 softmax 映射到一个 program。
6. [06_flash_attention_v2.md](06_flash_attention_v2.md) /
   [06_flash_attention_v2.py](06_flash_attention_v2.py)
   - 用 Triton 写 FlashAttention-2 风格的 forward kernel, 重点是 work partitioning。
7. [fa4/README.md](fa4/README.md)
   - 只保留 6 个核心文件的 FlashAttention-4 极简阅读版, 适合先抓主线。

## 运行方式

```bash
python learn/flash_attention/01_softmax.py
python learn/flash_attention/02_online_softmax.py
python learn/flash_attention/03_attention_baseline.py
python learn/flash_attention/04_flash_attention_v1.py
python learn/flash_attention/05_triton_softmax.py
python learn/flash_attention/06_flash_attention_v2.py
```

这些脚本需要 CUDA 和 Triton。没有 CUDA/Triton 时不会退回 PyTorch 实现,
因为本目录的代码目标就是展示 Triton 写法。

## 主线记忆

普通 attention 做的是:

```text
S = Q K^T / sqrt(d)
P = softmax(S)
O = P V
```

FlashAttention 没有改变数学结果。它改变的是执行顺序:

```text
for Q block:
    keep row max m, row sum l, output accumulator acc in registers/SRAM
    for K,V block:
        update m, l, acc with online softmax
    write final O block once
```

所以它是 exact attention, 不是 sparse attention, low-rank attention, 或近似 attention。

## 参考资料

- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
- Triton fused attention tutorial:
  https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
