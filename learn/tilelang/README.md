# TileLang 学习实验

这个目录放基于 `tilelang` 的学习脚本。

## 文件

- `flashatt.py`：实现一个 TileLang 版 FlashAttention forward，并和 PyTorch `scaled_dot_product_attention` 做正确性与性能对比。

## 运行

```bash
python learn/tilelang/flashatt.py
```

## 当前实现说明

- 输入布局使用 `BHSD`：`[batch, heads, seq, dim]`
- 只做 forward，对比对象是 PyTorch SDPA
- 当前脚本固定 `float16`
- `seq_kv >= seq_q`，兼容普通 attention 和带 `past_len` 的 causal attention

## 备注

- 仓库约束下不使用 `argparse`，改参数请直接修改 `flashatt.py` 里的 `FlashAttConfig`
- 非方阵 causal 场景下，参考实现使用 PyTorch 的 `causal_lower_right`，与 TileLang 官方示例的掩码语义保持一致
