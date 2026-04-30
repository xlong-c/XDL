# 05. Triton Fused Softmax

前面已经用 Triton 写了 safe softmax 和 online softmax。这里再单独整理一个
更标准的 fused row softmax kernel, 方便和 Triton 官方教程的写法对齐。

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
多个 op 之间反复读写全量 tensor。

## 为什么先写 softmax kernel

FlashAttention 的 forward kernel 本质上就是:

```text
QK matmul tile
row softmax update
PV matmul tile
```

如果看懂了 Triton softmax 的 row program, 再看 FlashAttention 的 `Q block`
program 会容易很多。

## 运行

```bash
python learn/flash_attention/05_triton_softmax.py
```

脚本需要 CUDA 和 Triton。PyTorch 只用于创建输入和检查误差。
