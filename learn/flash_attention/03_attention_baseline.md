# 03. Baseline Attention

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

问题在 `S` 和 `P`。它们都是 `N x N`, 序列越长越贵。

例如 `batch=1, heads=32, seq_len=8192, dtype=float16`:

```text
S + P = 1 * 32 * 8192 * 8192 * 2 tensors * 2 bytes
      = 8 GiB
```

还没算 Q, K, V, O 和 backward 需要的中间量。

## FlashAttention 要解决什么

FlashAttention 不改变 `O = softmax(QK^T)V` 的结果。它只是避免把完整的
`S` 和 `P` 写到 HBM, 改成按 block 计算并用 online softmax 合并。

## 本节代码

`03_attention_baseline.py` 故意用 Triton 分三步做最朴素的 attention:

1. `_scores_kernel` 写出完整 `S = QK^T / sqrt(d)`。
2. `_row_softmax_kernel` 写出完整 `P = softmax(S)`。
3. `_pv_kernel` 读取 `P` 和 `V`, 写出 `O = PV`。

这不是高效实现, 目的是用 Triton 看清楚 baseline 的 HBM 写回成本。

## 运行

```bash
python learn/flash_attention/03_attention_baseline.py
```
