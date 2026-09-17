# xdl-jax 项目规范

## 目录职责

- 承载 `xdl-jax` 的正式实现,测试,示例和项目文档.
- 保持 JAX 训练 runtime 的实现边界和可验证交付状态.
- 研究结论统一引用 `../research/xdl-jax/RESEARCH.md`.

本项目不承载:

- JAX 训练数据本体.
- XQT 量化 kernel, 导出实现或 serving runtime.
- XDL PyTorch 主链路的 Torch 类型和 Trainer 实现.

## 文档分工

- `README.md`: 项目入口,安装和当前状态.
- `GUIDE.md`: 长期指导, 约束设计和实现方向.
- `docs/`: 长期使用和 API 文档.
- `research/xdl-jax/RESEARCH.md`: 调研事实和来源,不写未验证的实现状态.
- `DESIGN.md`: 设计方案和 API 草案, 可以包含尚未实现的接口.
- `TODO.md`: 可执行任务和验收记录, 不替代设计文档.
- `xdl_jax/`: 可安装的实现包.
- `tests/`: correctness, contract 和发布 smoke 测试.
- `examples/`: 可复跑的 CPU/GPU smoke 和配置示例.

调研事实, 设计决策和执行状态必须分开. 不要在 `TODO.md` 中重复完整架构设计, 也不要把未实现的设计写成 `RESEARCH.md` 中的既成事实.

## 工作区边界

`xdl-jax` 属于 XDL 训练侧:

```text
PyTorch 训练 -> XDL
JAX 训练     -> xdl-jax
模型压缩部署 -> XQT
```

不得因为 JAX 生态包含 MaxText, Tunix, inference engine 或量化库, 就把这些职责纳入 `xdl-jax`.

特别禁止:

- 在 `xdl-jax` 中新增训练之外的量化/部署 registry.
- 在 `xqt` 中新增训练循环, `JaxTrainer` 或 post-training provider.
- 把 `torch.Tensor`, `torch.nn.Module`, `torch.optim.Optimizer` 作为 `xdl-jax` 核心 API 类型.
- 把当前 PyTorch `CoreModel.manual_backward()` 语义直接复制到 JAX.

## 实现规则

- 新增长期事实前先核对当前 XDL 的架构文档和相关源码.
- 第三方库的 API, 版本, 安装方式和推荐实践必须以官方文档为准.
- JAX 分布式行为必须通过实际 smoke 或官方文档确认, 不根据 `pmap` 旧经验猜测.
- 所有公共接口都要写清楚输入 PyTree, 输出 PyTree, RNG, shape, dtype 和 sharding 语义.
- 所有 checkpoint 方案都要写清楚 model, optimizer, RNG, data iterator 和 metadata 是否保存.
- 不能用 benchmark 的 planned 或 metadata-only 结果宣称 native 能力.
- 示例和验证脚本必须服务于 `xdl-jax` 的可复跑入口,不能变成平行训练框架.

## 代码落地后的同步规则

代码实现阶段:

1. 先更新 `TODO.md` 对应任务和验收门槛.
2. 实现后补测试或运行证据.
3. 公开 API 变化同步 `DESIGN.md` 和项目 API 边界文档.
4. 配置字段变化同步 schema, 示例和配置文档.
5. 结构性变更后刷新 `xdl` 知识图谱.
6. 中文文档或注释使用半角标点, 完成后运行:

```bash
XDL_PUNCT_PATHS=xdl-jax python scripts/normalize_punctuation.py
```

只检查时使用:

```bash
XDL_PUNCT_CHECK=1 XDL_PUNCT_PATHS=xdl-jax python scripts/normalize_punctuation.py
```

## 验收口径

以下内容不能单独作为完成证据:

- 仅创建了空模块.
- 仅能 import.
- 仅有接口占位符.
- 仅有 metadata 或 planned registry entry.
- 仅有 synthetic benchmark, 没有 correctness.

一个实现任务至少需要对应的测试, smoke, artifact, report 或明确的阻塞记录.
