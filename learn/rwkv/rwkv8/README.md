# RWKV v8（Heron + ROSA）技术整理

> 更新时间：2026-03-03（美国时间）  
> 目标：汇总 RWKV v8 当前公开实现的关键资讯、核心技术细节与关键架构代码（精简版）

---

## 1. 快速结论（先看这个）

1. **RWKV v8 目前仍是实验形态**，官方公开内容以 `RWKV-v8/` 目录脚本为主（不是单一稳定论文实现）。
2. v8 的核心新点是 **ROSA（Rapid Online Suffix Automaton）**：用后缀匹配机制做符号级记忆检索。
3. 当前公开实现包含两条路线：
   - **纯 ROSA + FFN**（`260212_rosa1bitLM_L12.py`、`260222_rosa4bitLM_L12.py`）
   - **RWKV7 + ROSA 混合**（`251024_rosaQKV_run.py`、`251105_reverse_run.py`）
4. 官方脚本里存在明显原型特征：不少 ROSA 路径仍是 Python/CPU 参考实现，推理也有全上下文重算示例（速度慢）。

---

## 2. 关键资讯（时间线 + 状态）

### 2.1 官方信息

- 官方仓库：`https://github.com/BlinkDL/RWKV-LM`
- v8 方向说明：`RWKV-8.md`（偏研究框架，不是完整训练规范）
- v8 目录说明：`RWKV-v8/README.md`（给了 demo、结果图、社区项目入口）

### 2.2 可定位到的关键时间点

- **2026-02-12**：`260212_rosa1bitLM_L12.py`（ROSA-1bit L12 demo）
- **2026-02-21**：`260222_rosa4bitLM_L12.py`（ROSA-4bit L12 demo）
- 本次检索的官方仓库快照：`main` 分支 `844a910`（提交时间 2026-02-21）

### 2.3 当前定位

- 截至 2026-03-03，官方 RWKV v8 对外公开材料更接近 **“快速迭代的实现集合”**，而不是一个“冻结规范”的单一架构版本。

---

## 3. 技术核心：ROSA 在做什么？

ROSA 可以理解为：

- 输入三路序列：`q`（query）、`k`（key）、`v`（value）
- 对每个时刻 `t`，在历史 `k` 中找与当前 `q` 末尾最匹配的后缀片段
- 若匹配到位置 `(j, w)`，输出“后继位置”的 `v[j+w]`
- 若未匹配，输出默认值（不同脚本有不同处理）

### 3.1 一种等价描述

对每个 `t`：

- 令 `S_t = q[0:t]` 的所有后缀集合
- 在历史 `k[0:t-1]` 中寻找最长可匹配后缀
- 返回匹配结束后一个位置对应的 `v` 符号

这是一种**符号匹配记忆**，不是标准 attention 的加权和。

---

## 4. 代码级实现细节（官方脚本抽取）

> 说明：以下代码是按官方实现逻辑重写的“精简结构版”，便于阅读与复现，不是逐行拷贝。

### 4.1 ROSA 1-bit 层（核心数据流）

```python
import torch
import torch.nn as nn

def rosa_batch_ref(qb: torch.Tensor, kb: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """
    qb/kb/vb: [B*C, T], uint8
    返回 idx: [B*C, T], uint8（0/1）
    内部逻辑对应在线后缀自动机匹配（Python 参考实现）
    """
    # 这里省略 SAM 细节，见官方 rosa_qkv_ref / samx_qkv_slow
    raise NotImplementedError


class RosaQKV1bitOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, emb: torch.Tensor):
        # q/k/v: [B, T, C]
        B, T, C = q.shape
        qb = (q > 0).to(torch.uint8).transpose(1, 2).reshape(-1, T).contiguous()
        kb = (k > 0).to(torch.uint8).transpose(1, 2).reshape(-1, T).contiguous()
        vb = (v > 0).to(torch.uint8).transpose(1, 2).reshape(-1, T).contiguous()

        idx = rosa_batch_ref(qb, kb, vb).view(B, C, T).transpose(1, 2).contiguous()
        out = (2.0 * idx.to(q.dtype) - 1.0) * emb  # 0/1 -> -1/+1，再乘通道幅度
        return out


class RosaQKV1bit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.emb = nn.Parameter(torch.ones(1, 1, channels))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return RosaQKV1bitOp.apply(q, k, v, self.emb)
```

关键点：

- 先用 `sign(q/k/v)` 做二值化，再走 ROSA 匹配；
- 按 `(B, C)` 展开后逐通道处理时序；
- 输出是符号结果映射到 `±emb`，`emb` 为可学习通道尺度。

### 4.2 ROSA 4-bit 层（分组符号化）

```python
class RosaQKV4bit(nn.Module):
    def __init__(self, channels: int, bits_per_symbol: int = 4):
        super().__init__()
        self.bits = bits_per_symbol
        self.emb = nn.Parameter(torch.ones(1, 1, channels))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # 1) 每 bits 个通道打包成一个离散 symbol
        # 2) 对 symbol 序列做 ROSA 匹配
        # 3) 命中时把 symbol 逐 bit 展开回通道，映射到 ±emb
        # 4) 未命中置 0（官方 4bit demo 是这个语义）
        raise NotImplementedError
```

关键点：

- 4bit 版本把表达容量从二值符号扩展到 16 个符号；
- 官方 4bit 脚本中，未匹配输出显式为 0。

---

## 5. 关键架构代码（精简版）

### 5.1 纯 ROSA + FFN Block（对应 260212/260222 思路）

```python
class RosaMix(nn.Module):
    def __init__(self, c: int, rosa_layer: nn.Module):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_q = nn.Parameter(torch.zeros(1, 1, c))
        self.x_k = nn.Parameter(torch.zeros(1, 1, c))
        self.x_v = nn.Parameter(torch.zeros(1, 1, c))
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(c, c)
        self.v = nn.Linear(c, c)
        self.rosa = rosa_layer
        self.out = nn.Linear(c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        q = x + xx * self.x_q
        k = x + xx * self.x_k
        v = x + xx * self.x_v
        y = self.rosa(self.q(q), self.k(k), self.v(v))
        return self.out(y)


class FFN(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, c))
        self.key = nn.Linear(c, 4 * c, bias=False)
        self.val = nn.Linear(4 * c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        return self.val(torch.relu(self.key(k)) ** 2)


class PureRosaBlock(nn.Module):
    def __init__(self, c: int, rosa_layer: nn.Module):
        super().__init__()
        self.ln_rosa = nn.LayerNorm(c)
        self.ln_ffn = nn.LayerNorm(c)
        self.rosa = RosaMix(c, rosa_layer)
        self.ffn = FFN(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.rosa(self.ln_rosa(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x
```

### 5.2 RWKV7 + ROSA 混合 Block（对应 251024/251105 思路）

```python
class HybridBlock(nn.Module):
    def __init__(self, c: int, rwkv7_tmix: nn.Module, rosa: nn.Module, ffn: nn.Module):
        super().__init__()
        self.ln_tmix = nn.LayerNorm(c)
        self.ln_rosa = nn.LayerNorm(c)
        self.ln_ffn = nn.LayerNorm(c)
        self.tmix = rwkv7_tmix
        self.rosa = rosa
        self.ffn = ffn

    def forward(self, x: torch.Tensor, v_first: torch.Tensor):
        xr = self.rosa(self.ln_rosa(x))
        xt, v_first = self.tmix(self.ln_tmix(x), v_first)
        x = x + xr + xt
        x = x + self.ffn(self.ln_ffn(x))
        return x, v_first
```

> 混合路线中，RWKV7 的 `Tmix` 仍调用 `wkv7` CUDA 算子（`wind_backstepping`）；ROSA 分支负责符号匹配记忆。

---

## 6. RWKV7 分支里值得注意的实现细节

在 `RWKV_Tmix_x070`（混合脚本）中可看到：

1. **time-shift 混合输入**：`xr/xw/xk/xv/xa/xg` 都由 `x + (shift(x)-x)*x_*` 形成。  
2. **动态衰减与门控**：`w/a/v/g` 等参数通过低秩因子（LoRA-like）动态生成。  
3. **数值稳定约束**：`w` 通过 `softplus` 变换并限制到较稳定范围。  
4. **头内归一化**：`kk` 进行按 head 的 `L2 normalize`。  
5. **CUDA 核调用**：`RUN_CUDA_RWKV7g(...)`，再经 `GroupNorm` + 输出线性层。

---

## 7. 目前公开实现的工程现状与挑战

1. **ROSA 主干仍偏参考实现**：Python/CPU 路径多，长序列训练或推理吞吐受限。  
2. **增量解码未完全工程化**：官方示例里存在每步全上下文重算（明确标注 very slow）。  
3. **架构仍在快速迭代**：脚本命名和实验任务（算术、反转、copy/count）体现探索阶段。  
4. **文档形态偏分散**：需要结合多个脚本才能看清完整设计。

---

## 8. 复现实操建议（最小闭环）

1. 先跑官方 demo（小规模验证）：

```bash
git clone https://github.com/BlinkDL/RWKV-LM
cd RWKV-LM
python RWKV-v8/260212_rosa1bitLM_L12.py
python RWKV-v8/260222_rosa4bitLM_L12.py
```

2. 再看混合版：

```bash
python RWKV-v8/251024_rosaQKV_run.py
python RWKV-v8/251105_reverse_run.py
```

3. 若要做工程化：

- 优先把 ROSA 算子迁移到 C++/CUDA（至少先消除 Python 循环）；
- 补齐增量状态缓存，避免解码时全量重算；
- 把 1-bit / 4-bit 路径统一成可配置算子接口。

---

## 9. 资料索引（建议长期跟踪）

### 官方

- `https://github.com/BlinkDL/RWKV-LM`
- `https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-8.md`
- `https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/README.md`
- `https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/260212_rosa1bitLM_L12.py`
- `https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/260222_rosa4bitLM_L12.py`
- `https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251024_rosaQKV_run.py`
- `https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251105_reverse_run.py`

### 社区（实现与论文化）

- `https://github.com/johanwind/wind_rosa`
- `https://github.com/wjie98/rosa_soft`
- `https://github.com/zyaaa-ux/ROSA-Tuning`
- `https://arxiv.org/abs/2602.02499`

---

## 10. 本文与源码的对应关系

- ROSA 1bit 逻辑：对齐 `rosa_qkv_ref` / `rosa_qkv_1bit_layer_op` / `RWKV_ROSA_1bit`
- ROSA 4bit 逻辑：对齐 `rosa_slow_4bit_layer` / `RWKV_ROSA_4bit`
- 混合块：对齐 `ROSA_QKV_B_1bit` + `RWKV_Tmix_x070` + `FFN`
- 纯 ROSA 块：对齐 L12 demo 里的 `Block` 组织方式

如果后续你要把这份文档升级为“可训练实现文档”，建议下一步新增：

1. 统一的 `RosaOp` 接口（1bit/4bit 插拔）
2. 增量解码状态定义（ROSA 状态 + RWKV 状态）
3. 训练超参数模板（ctx、lr、warmup、loss、tokenizer）

