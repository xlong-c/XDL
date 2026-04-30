# 01. Softmax

Softmax 把一行 logits 变成概率:

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

直接算 `exp(x)` 有两个问题:

- `x` 很大时会 overflow, 例如 `exp(1000)`。
- `x` 很小时会 underflow, 数值会变成 0。

常用写法是减去这一行的最大值:

```text
m = max(x)
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

这样不会改变结果, 因为分子分母同时除以了 `exp(m)`。

## 和 attention 的关系

标准 attention 的每一行都是一个 softmax:

```text
scores[row, :] = q[row] @ K^T / sqrt(d)
probs[row, :] = softmax(scores[row, :])
out[row] = probs[row, :] @ V
```

后面的 FlashAttention 只是在避免把完整的 `scores` 和 `probs` 写到 HBM。
数学仍然是这一行 softmax。

## 本节代码

`01_softmax.py` 里有两个 Triton kernel:

- `_naive_softmax_kernel`: 直接 `exp(x)`, 会在大 logits 上溢出。
- `_safe_softmax_kernel`: 先减 row max, 同时把 `m` 和 `l` 写出来。

PyTorch 只用于生成输入和做参考答案。

## 运行

```bash
python learn/flash_attention/01_softmax.py
```
