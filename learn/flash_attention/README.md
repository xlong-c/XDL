# Softmax to FlashAttention-2

这个目录是一条教学路线, 目标是从最普通的 `softmax` 一直推到
FlashAttention-2 的 forward kernel 思路.讲义已收拢到当前 `README.md`,
对应代码仍保留为独立 `.py` 文件.所有主实现都写成 Triton kernel; PyTorch 只用于创建 tensor
和做 correctness reference.

这不是生产级替代品.这里的 Triton kernel 只覆盖教学所需的 forward 路径,
没有 backward, dropout, ALiBi, varlen, autotune, FP8 等工程细节.

## 学习顺序

1. `01_softmax.py`: 从 naive softmax 到 safe softmax, 理解为什么要减去 row max.
2. `02_online_softmax.py`: 把 softmax 的分母改成 streaming 更新, 这是 FlashAttention 的核心小积木.
3. `03_attention_baseline.py`: 用 Triton 显式 materialize `N x N` logits/probs, 看清楚为什么吃显存.
4. `04_flash_attention_v1.py`: 用 Triton block loop 写一个精确的 FlashAttention v1 教学版.
5. `05_triton_softmax.py`: 用 Triton 写 fused row softmax, 把一行 softmax 映射到一个 program.
6. `06_flash_attention_v2.py`: 用 Triton 写 FlashAttention-2 风格的 forward kernel, 重点是 work partitioning.
7. [fa4/README.md](fa4/README.md): 只保留 6 个核心文件的 FlashAttention-4 极简阅读版, 适合先抓主线.

## 运行方式

```bash
python learn/flash_attention/01_softmax.py
python learn/flash_attention/02_online_softmax.py
python learn/flash_attention/03_attention_baseline.py
python learn/flash_attention/04_flash_attention_v1.py
python learn/flash_attention/05_triton_softmax.py
python learn/flash_attention/06_flash_attention_v2.py
```

这些脚本需要 CUDA 和 Triton.没有 CUDA/Triton 时不会退回 PyTorch 实现,
因为本目录的代码目标就是展示 Triton 写法.

## 主线记忆

普通 attention 做的是:

```text
S = Q K^T / sqrt(d)
P = softmax(S)
O = P V
```

FlashAttention 没有改变数学结果.它改变的是执行顺序:

```text
for Q block:
    keep row max m, row sum l, output accumulator acc in registers/SRAM
    for K,V block:
        update m, l, acc with online softmax
    write final O block once
```

所以它是 exact attention, 不是 sparse attention, low-rank attention, 或近似 attention.

## 01. Softmax

Softmax 把一行 logits 变成概率:

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

直接算 `exp(x)` 有两个问题:

- `x` 很大时会 overflow, 例如 `exp(1000)`.
- `x` 很小时会 underflow, 数值会变成 0.

常用写法是减去这一行的最大值:

```text
m = max(x)
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

这样不会改变结果, 因为分子分母同时除以了 `exp(m)`.

### 和 attention 的关系

标准 attention 的每一行都是一个 softmax:

```text
scores[row, :] = q[row] @ K^T / sqrt(d)
probs[row, :] = softmax(scores[row, :])
out[row] = probs[row, :] @ V
```

后面的 FlashAttention 只是在避免把完整的 `scores` 和 `probs` 写到 HBM.
数学仍然是这一行 softmax.

### 本节代码

`01_softmax.py` 里有两个 Triton kernel:

- `_naive_softmax_kernel`: 直接 `exp(x)`, 会在大 logits 上溢出.
- `_safe_softmax_kernel`: 先减 row max, 同时把 `m` 和 `l` 写出来.

PyTorch 只用于生成输入和做参考答案.

### 运行

```bash
python learn/flash_attention/01_softmax.py
```

## 02. Online Softmax

Safe softmax 需要先知道整行最大值 `m`, 再算分母 `l`:

```text
m = max(x)
l = sum_j exp(x_j - m)
softmax(x_i) = exp(x_i - m) / l
```

FlashAttention 的问题是: 一整行 scores 可能太长, 我们不想把它全部放在内存里.
所以要把 `m` 和 `l` 改成可以分块更新的状态.

### 分块更新公式

已有旧状态:

```text
m_old = max(old blocks)
l_old = sum exp(x_old - m_old)
```

来了一个新 block:

```text
m_block = max(block)
l_block = sum exp(block - m_block)
```

合并:

```text
m_new = max(m_old, m_block)
l_new = exp(m_old - m_new) * l_old
      + exp(m_block - m_new) * l_block
```

这就是 online softmax.它保留精确结果, 但只需要保存两个标量状态.

### 对 attention 的启发

在 attention 里每个 query row 不只要分母, 还要输出:

```text
out = sum_j softmax(score_j) * V_j
```

所以 FlashAttention 保存的状态是:

```text
m: row max
l: shifted denominator
acc: shifted weighted sum over V
```

### 本节代码

`02_online_softmax.py` 用一个 Triton program 处理一行 logits.kernel 内部先
按 `BLOCK_N` 扫描并在线合并 `(m, l)`, 再用最终的 `(m, l)` 写回这一行概率.

这个两遍写法是教学版: 它展示 online softmax 状态如何在 Triton 里更新.
FlashAttention 后面会把第二遍写概率改成直接累加 `P @ V`, 从而不写出完整
`P`.

### 运行

```bash
python learn/flash_attention/02_online_softmax.py
```

## 03. Baseline Attention

Scaled dot-product attention 是:

```text
S = Q K^T / sqrt(d)
P = softmax(S)
O = P V
```

形状通常是:

```text
Q, K, V: [batch, heads, seq_len, head_dim]
S, P:    [batch, heads, seq_len, seq_len]
O:       [batch, heads, seq_len, head_dim]
```

问题在 `S` 和 `P`.它们都是 `N x N`, 序列越长越贵.

例如 `batch=1, heads=32, seq_len=8192, dtype=float16`:

```text
S + P = 1 * 32 * 8192 * 8192 * 2 tensors * 2 bytes
      = 8 GiB
```

还没算 Q, K, V, O 和 backward 需要的中间量.

### FlashAttention 要解决什么

FlashAttention 不改变 `O = softmax(QK^T)V` 的结果.它只是避免把完整的
`S` 和 `P` 写到 HBM, 改成按 block 计算并用 online softmax 合并.

### 本节代码

`03_attention_baseline.py` 故意用 Triton 分三步做最朴素的 attention:

1. `_scores_kernel` 写出完整 `S = QK^T / sqrt(d)`.
2. `_row_softmax_kernel` 写出完整 `P = softmax(S)`.
3. `_pv_kernel` 读取 `P` 和 `V`, 写出 `O = PV`.

这不是高效实现, 目的是用 Triton 看清楚 baseline 的 HBM 写回成本.

### 运行

```bash
python learn/flash_attention/03_attention_baseline.py
```

## 04. FlashAttention v1

FlashAttention v1 的关键不是近似, 而是 IO-aware tiling.

普通 attention 会把两张大矩阵写到 HBM:

```text
S = QK^T
P = softmax(S)
```

FlashAttention 改成:

```text
for each Q block:
    m = -inf
    l = 0
    acc = 0
    for each K,V block:
        scores = Q_block @ K_block^T
        use online softmax to update m, l, acc
    O_block = acc / l
```

这里 `m`, `l`, `acc` 都是按 query row 保存的状态.

### 正确性重点

错误的教学实现常见问题是: 第 `i` 个 Q block 只看第 `i` 个 K,V block.
那不是 attention, 因为每个 query 都应该 attend 到所有可见 key.

本节代码里每个 Q block 都会扫描完整 K,V 序列, 只是不会 materialize 完整
`seq_len x seq_len` 的 scores/probs.

### 本节代码

`04_flash_attention_v1.py` 已经是 Triton kernel, 不是 PyTorch loop.每个
Triton program 负责一个 `(batch, head, Q block)`, 在 kernel 内循环扫描
所有 K,V block, 用 online softmax 更新 `m`, `l`, `acc`, 最后只写一次
`O block`.

### v1 的局限

v1 已经大幅减少 HBM IO, 但 GPU work partitioning 还不够理想:

- 非 matmul FLOPs 仍然偏多.
- 当 `batch * heads` 较小时, thread blocks 不够多, occupancy 可能不足.
- block 内 warp 分工会带来额外 shared memory 通信.

这些就是 FlashAttention-2 要改的地方.

### 运行

```bash
python learn/flash_attention/04_flash_attention_v1.py
```

## 05. Triton Fused Softmax

前面已经用 Triton 写了 safe softmax 和 online softmax.这里再单独整理一个
更标准的 fused row softmax kernel, 方便和 Triton 官方教程的写法对齐.

Triton 的基本映射:

```text
one program = one row
tl.arange   = columns inside that row
tl.load     = read row
tl.max      = row max
tl.exp      = numerator
tl.sum      = denominator
tl.store    = write row
```

这就是 fused softmax: max, exp, sum, divide 都在一个 kernel 里完成, 不需要
多个 op 之间反复读写全量 tensor.

### 为什么先写 softmax kernel

FlashAttention 的 forward kernel 本质上就是:

```text
QK matmul tile
row softmax update
PV matmul tile
```

如果看懂了 Triton softmax 的 row program, 再看 FlashAttention 的 `Q block`
program 会容易很多.

### 运行

```bash
python learn/flash_attention/05_triton_softmax.py
```

脚本需要 CUDA 和 Triton.PyTorch 只用于创建输入和检查误差.

## 06. FlashAttention-2

FlashAttention v1 解决了主要 IO 问题, 但还没有完全接近 GEMM 的效率.
FlashAttention-2 的论文把问题拆成三类:

- 减少 non-matmul FLOPs.
- 增加并行度, 尤其是 `batch * heads` 较小时也要有足够 thread blocks.
- 改善 block 内 warp 的 work partitioning, 减少 shared memory 往返.

本节代码实现一个教学版 Triton forward kernel.它保留 FlashAttention 的 exact
online softmax, 并采用 FA2 风格的 launch 方式:

```text
grid x: Q blocks along sequence
grid y: batch * heads

each program:
    owns one Q block
    loops over all K,V blocks
    keeps m, l, acc on chip
    writes one O block
```

这和 Triton 官方 fused attention tutorial 的 forward 主干一致.为了教学可读性,
这里没有实现 production FA2 的 backward, dropout, autotune, varlen, FP8,
以及更细的 warp specialization.

### 和 v1 教学版的差别

`04_flash_attention_v1.py` 已经是 Triton online attention kernel.
`06_flash_attention_v2.py` 保持同样的 exact attention 数学, 但用 FA2 的视角
强调 launch grid,work partitioning,减少非 matmul 工作和提高 occupancy.

### 运行

```bash
python learn/flash_attention/06_flash_attention_v2.py
```

脚本需要 CUDA 和 Triton.PyTorch 只用于 correctness reference.

## 参考资料

- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
- Triton fused attention tutorial:
  https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
