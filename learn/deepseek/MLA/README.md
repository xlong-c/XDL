# DeepSeek MLA (Multi-Head Latent Attention)

## 概述

这个文件夹实现了 DeepSeek 的 MLA (Multi-Head Latent Attention) 注意力机制，并与标准的多头注意力 (MHA) 进行了对比。

## 文件说明

- `mha.py` - 标准 Multi-Head Attention 实现作为对比基线
- `mla.py` - DeepSeek 的 Multi-Head Latent Attention 实现
- `compare.py` - MHA vs MLA 对比实验脚本
- `*.png` - 实验结果可视化图表

## 核心概念

### MHA (Multi-Head Attention)

**特点:**
- 存储完整的 K、V 矩阵
- KV Cache 大小: `O(seq_len * num_heads * head_dim * 2)`
- 实现简单，无额外计算开销
- 长序列时内存占用巨大

### MLA (Multi-Head Latent Attention)

**核心思想:**
1. 使用低维度的 latent key 和 latent value 来压缩 KV cache
2. 只存储压缩后的 KV，大幅减少内存占用
3. 通过 up-projection 恢复到所需的维度进行注意力计算

**特点:**
- KV Cache 大小: `O(seq_len * latent_dim * 2)`
- 压缩率通常为 8-12 倍
- 内存节省 80-90%
- 增加了上投影的计算开销

## 快速开始

### 1. 测试 MHA

```bash
cd learn/MLA
python mha.py
```

### 2. 测试 MLA

```bash
cd learn/MLA
python mla.py
```

### 3. 运行完整对比实验

```bash
cd learn/MLA
python compare.py
```

## 实验结果

### 实验 1: 内存占用对比

展示了不同序列长度下 MHA 和 MLA 的 KV Cache 内存占用：

| 序列长度 | MHA (MB) | MLA (MB) | 节省 |
|---------|----------|----------|------|
| 512     | 20.00    | 2.00     | 90.0% |
| 1024    | 40.00    | 4.00     | 90.0% |
| 2048    | 80.00    | 8.00     | 90.0% |
| 4096    | 160.00   | 16.00    | 90.0% |
| 8192    | 320.00   | 32.00    | 90.0% |
| 16384   | 640.00   | 64.00    | 90.0% |

### 实验 2: 不同压缩率的效果

展示了不同压缩率对内存节省的影响：

| 压缩率 | Latent Dim | 节省 |
|--------|------------|------|
| 2x     | 2560       | 50.0% |
| 4x     | 1280       | 75.0% |
| 6x     | 853        | 83.3% |
| 8x     | 640        | 87.5% |
| 10x    | 512        | 90.0% |
| 12x    | 426        | 91.7% |
| 16x    | 320        | 93.8% |

### 实验 3: 推理速度对比

在相同配置下，MLA 的推理速度与 MHA 相当或略快：

- MHA: 6.18 ms
- MLA: 5.84 ms
- 差异: -5.5%

### 实验 4: 自回归生成中的 KV Cache 累积

展示了在自回归生成过程中，MLA 持续保持 87.5% 的内存节省。

## 适用场景

### MHA
- 短序列任务
- 对推理速度要求极高
- 内存充足的场景

### MLA
- 长序列任务 (如大语言模型)
- 内存受限的场景
- 需要支持超长上下文的模型

## DeepSeek 的选择

DeepSeek-V2/V3 使用 MLA 来支持超长上下文：
- 压缩率通常设置为 8-12 倍
- 在性能和效率之间取得了良好平衡
- 使得模型能够处理更长的输入序列

## 技术细节

### MLA 架构

```python
# 压缩
k_latent = W_down_k(x)  # [batch, seq_len, latent_dim]
v_latent = W_down_v(x)  # [batch, seq_len, latent_dim]

# 存储 (只存储压缩后的 KV)
kv_compressed = concat(k_latent, v_latent)  # [batch, seq_len, latent_dim * 2]

# 解压 (用于注意力计算)
k = W_up_k(k_latent)  # [batch, seq_len, num_heads * head_dim]
v = W_up_v(v_latent)  # [batch, seq_len, num_heads * head_dim]
```

### 内存占用计算

- MHA: `2 * batch_size * num_heads * seq_len * head_dim * 4` bytes
- MLA: `2 * batch_size * seq_len * latent_dim * 4` bytes

压缩率: `(num_heads * head_dim) / latent_dim`

## 参考

- DeepSeek-V2 Technical Report: https://arxiv.org/abs/2405.04434
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437

## 作者

基于 DeepSeek 的 MLA 论文实现
