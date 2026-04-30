# 06. FlashAttention-2

FlashAttention v1 解决了主要 IO 问题, 但还没有完全接近 GEMM 的效率。
FlashAttention-2 的论文把问题拆成三类:

- 减少 non-matmul FLOPs。
- 增加并行度, 尤其是 `batch * heads` 较小时也要有足够 thread blocks。
- 改善 block 内 warp 的 work partitioning, 减少 shared memory 往返。

本节代码实现一个教学版 Triton forward kernel。它保留 FlashAttention 的 exact
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

这和 Triton 官方 fused attention tutorial 的 forward 主干一致。为了教学可读性,
这里没有实现 production FA2 的 backward, dropout, autotune, varlen, FP8,
以及更细的 warp specialization。

## 和 v1 教学版的差别

`04_flash_attention_v1.py` 已经是 Triton online attention kernel。  
`06_flash_attention_v2.py` 保持同样的 exact attention 数学, 但用 FA2 的视角
强调 launch grid、work partitioning、减少非 matmul 工作和提高 occupancy。

## 运行

```bash
python learn/flash_attention/06_flash_attention_v2.py
```

脚本需要 CUDA 和 Triton。PyTorch 只用于 correctness reference。
