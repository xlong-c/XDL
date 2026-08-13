# xdl-jax 架构

`xdl-jax/` 是 XDL 面向 JAX 生态的独立训练项目. 它与当前 PyTorch
`xdl/` 并行,不把 JAX 类型塞入 Torch `CoreModel`/`Trainer` 主链.

## 职责边界

```text
xdl       -> PyTorch 训练
xdl-jax   -> JAX 训练编排
xqt       -> 模型压缩,图变换,导出和 benchmark
```

`xdl-jax` 负责 `JaxTask`, model adapter, Optax update, validation,
callback, Orbax checkpoint 和模型侧 artifact export. 它不负责量化,剪枝,
QAT,部署 runtime,serving 或 XQT workflow.

## 当前正式候选能力

- CPU 单设备训练和验证.
- NVIDIA 单卡 GPU 训练,包括 FP32 和 bfloat16 smoke.
- Functional 和 Flax NNX model adapter.
- gradient accumulation,显式 RNG 和 exact/weights-only restore.
- OmegaConf YAML setup, callback 和 checkpoint 配置.
- 2-device CPU data parallel correctness 验证.
- 模型侧 NumPy artifact 和 dtype/sharding metadata.

## 当前明确限制

- 多 GPU,多主机和 TPU 没有当前正式验收证据.
- 多设备 sharded checkpoint 的跨拓扑恢复未交付.
- 默认 NumPy data source 不承诺精确 iterator resume.
- XQT 交接当前通过模型侧 artifact helper,不在训练过程中调用 XQT.

实现入口见 [`xdl-jax/README.md`](../../../xdl-jax/README.md), API 和运行文档
见 [`xdl-jax/docs/`](../../../xdl-jax/docs/),调研事实见
[`research/xdl-jax/RESEARCH.md`](../../../research/xdl-jax/RESEARCH.md).
