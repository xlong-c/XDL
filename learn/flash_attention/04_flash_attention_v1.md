# 04. FlashAttention v1

FlashAttention v1 的关键不是近似, 而是 IO-aware tiling。

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

这里 `m`, `l`, `acc` 都是按 query row 保存的状态。

## 正确性重点

错误的教学实现常见问题是: 第 `i` 个 Q block 只看第 `i` 个 K,V block。
那不是 attention, 因为每个 query 都应该 attend 到所有可见 key。

本节代码里每个 Q block 都会扫描完整 K,V 序列, 只是不会 materialize 完整
`seq_len x seq_len` 的 scores/probs。

## 本节代码

`04_flash_attention_v1.py` 已经是 Triton kernel, 不是 PyTorch loop。每个
Triton program 负责一个 `(batch, head, Q block)`, 在 kernel 内循环扫描
所有 K,V block, 用 online softmax 更新 `m`, `l`, `acc`, 最后只写一次
`O block`。

## v1 的局限

v1 已经大幅减少 HBM IO, 但 GPU work partitioning 还不够理想:

- 非 matmul FLOPs 仍然偏多。
- 当 `batch * heads` 较小时, thread blocks 不够多, occupancy 可能不足。
- block 内 warp 分工会带来额外 shared memory 通信。

这些就是 FlashAttention-2 要改的地方。

## 运行

```bash
python learn/flash_attention/04_flash_attention_v1.py
```
