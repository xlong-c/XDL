# xdl-jax

`xdl-jax` 是 XDL 面向 JAX 生态的独立训练项目. 它负责 JAX-native 的模型训练编排, 但不改变 XDL 和 XQT 的职责边界:

```text
XDL       -> PyTorch 训练
xdl-jax   -> JAX 训练
XQT       -> 模型压缩, 图变换, 导出适配, benchmark
```

三者通过 checkpoint, 模型权重, 量化产物或其他明确的模型侧产物衔接. `xdl-jax` 不负责模型压缩, 部署 runtime, serving scheduler 或 KV cache 管理.

代码,测试和示例已经位于一级项目目录 `xdl-jax/`. 调研资料保留在
[`research/xdl-jax/RESEARCH.md`](../research/xdl-jax/RESEARCH.md),只记录选型和生态事实.

## 文档分工

| 文件 | 角色 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 项目工作规则和文档维护约束 |
| [GUIDE.md](GUIDE.md) | 长期架构原则, 设计边界和实现指导 |
| [DESIGN.md](DESIGN.md) | `xdl-jax` 的模块, API, 状态, 配置和分布式设计草案 |
| [TODO.md](TODO.md) | 可执行任务, 阶段门槛, 验收标准和完成记录 |
| [docs/](docs/) | 安装, API, 配置, checkpoint, 分布式和限制说明 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 开发, 检查和提交约定 |
| [CHANGELOG.md](CHANGELOG.md) | 版本变更记录 |

## 推荐阅读顺序

首次接手:

1. `AGENTS.md`
2. `GUIDE.md`
3. `DESIGN.md`
4. `TODO.md`

准备实现:

1. `GUIDE.md`
2. `DESIGN.md` 对应章节
3. `TODO.md` 当前阶段
4. 仓库根目录和目标源码目录的 `AGENTS.md`

## 当前状态

当前状态是 `0.1.0 alpha / production candidate`. 已完成第一阶段的独立项目化和第二阶段的单设备生产候选闭环. 当前代码位于 `xdl_jax/`, 测试位于 `tests/`, 示例位于 `examples/`.

当前实现边界:

- 已验证: CPU/GPU 单设备, JIT train/eval, functional adapter, Flax NNX adapter, Optax, gradient accumulation, deterministic RNG, exact state restore, weights-only restore, strict YAML setup, callback ordering/error policy, 固定 shape batch validation, 单主机 2-device CPU data parallel, `shard_map` batch/state placement 和 collective reduction, compile/steady-state benchmark, 以及模型侧 artifact export.
- GPU 已验证: CUDA 13 JAX plugin, RTX 4070 Ti SUPER, FP32 和 bfloat16 synthetic regression training, GPU device placement 和 loss decrease.
- 未宣称支持: TPU 实测, 多主机, Grain, 多设备 sharded checkpoint 的跨拓扑恢复, XQT 输入交接和多 GPU 正式验收.

## 安装与验证

CPU:

```bash
python -m pip install -e '.[cpu,test]'
PYTHONPATH=. pytest tests -q
PYTHONPATH=. python examples/run_linear_regression.py
```

NVIDIA GPU + CUDA 12:

```bash
python -m pip install -e '.[gpu-cuda12,test]'
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. pytest tests/test_gpu.py -q
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. python examples/run_gpu_linear_regression.py
```

NVIDIA GPU + CUDA 13:

```bash
python -m pip install -e '.[gpu-cuda13,test]'
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. pytest tests/test_gpu.py -q
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. python examples/run_gpu_linear_regression.py
```

`runtime.platform` 或 `TrainerConfig.platform` 支持 `auto`, `cpu`, `gpu`,
`tpu`. 指定 `gpu` 时,缺少 CUDA-enabled JAX plugin 会显式失败,不会把 GPU
任务静默降级成 CPU. 多个训练进程共享一张 GPU 时,建议关闭 JAX 默认显存预分配,
并由调用方控制 `XLA_PYTHON_CLIENT_MEM_FRACTION`.

当前 tested combination:

- CPU: Python 3.12.12, JAX 0.11.0, Flax 0.12.8, Optax 0.2.8,
  Orbax Checkpoint 0.12.4.
- GPU: Python 3.12.12, JAX 0.11.0, `jax-cuda13-plugin` 0.11.0,
  `jax-cuda13-pjrt` 0.11.0, NVIDIA GeForce RTX 4070 Ti SUPER,
  driver 591.86, CUDA toolkit 13.1.

实现目录:

- `xdl_jax/types.py`: immutable state 和 PyTree contract.
- `xdl_jax/trainer.py`: CPU/GPU/TPU single-device compiled train/eval loop.
- `xdl_jax/model/`: functional 和 NNX adapter.
- `xdl_jax/checkpoint.py`: Orbax thin facade.
- `xdl_jax/distributed/`: 单设备和单主机一维 data parallel strategy, 包含 batch sharding, replicated state 和 `lax.pmean` reduction.
- `xdl_jax/performance.py`: compile 与 steady-state 分离的 benchmark report.

## 快速 API

```python
from xdl_jax import JaxTrainer, TrainerConfig
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData

result = JaxTrainer(
    LinearRegressionTask(input_dim=4),
    config=TrainerConfig(max_epochs=3, platform="cpu"),
).fit(
    SyntheticRegressionData(
        n_samples=128,
        input_dim=4,
        batch_size=16,
        seed=42,
    )
)
```

公共 API 只包括 `JaxTask`, `ModelAdapter`, `JaxTrainState`,
`JaxTrainer`, `CheckpointCallback`, `JaxCheckpointManager`,
`FunctionalModelAdapter` 和 `NNXModelAdapter` 等训练侧对象. 详细边界见
[`docs/API.md`](docs/API.md).

## 发布状态

- Python: `>=3.12`.
- CPU: CI 目标平台.
- NVIDIA GPU: CUDA 12/13 optional extra,当前机器已完成单卡 CUDA 13 smoke.
- TPU,多 GPU,多主机和跨拓扑 checkpoint: 保留为后续阶段,不会在没有实测证据时宣称支持.

开发和发布命令见 [`CONTRIBUTING.md`](CONTRIBUTING.md) 和
[`docs/RELEASE.md`](docs/RELEASE.md).

## 状态标记

TODO 中使用以下标记:

- `[ ]`: 未开始.
- `[>]`: 当前执行中.
- `[x]`: 已完成, 且有测试, 运行结果或文档证据.
- `[-]`: 明确不做或移出当前范围.

只有有证据的工作才能标记为 `[x]`. 设计完成不等于实现完成.
