# 02. Online Softmax

Safe softmax 需要先知道整行最大值 `m`, 再算分母 `l`:

```text
m = max(x)
l = sum_j exp(x_j - m)
softmax(x_i) = exp(x_i - m) / l
```

FlashAttention 的问题是: 一整行 scores 可能太长, 我们不想把它全部放在内存里。
所以要把 `m` 和 `l` 改成可以分块更新的状态。

## 分块更新公式

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

这就是 online softmax。它保留精确结果, 但只需要保存两个标量状态。

## 对 attention 的启发

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

## 本节代码

`02_online_softmax.py` 用一个 Triton program 处理一行 logits。kernel 内部先
按 `BLOCK_N` 扫描并在线合并 `(m, l)`, 再用最终的 `(m, l)` 写回这一行概率。

这个两遍写法是教学版: 它展示 online softmax 状态如何在 Triton 里更新。
FlashAttention 后面会把第二遍写概率改成直接累加 `P @ V`, 从而不写出完整
`P`。

## 运行

```bash
python learn/flash_attention/02_online_softmax.py
```
