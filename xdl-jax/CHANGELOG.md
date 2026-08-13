# Changelog

## 0.1.0 - 2026-08-13

首个正式一级项目版本.

- 提供 CPU 和单卡 GPU JAX 训练 runtime.
- 提供 Functional / Flax NNX model adapter.
- 提供 Optax optimizer, gradient accumulation 和 precision policy.
- 提供 OmegaConf setup, callback 和 Orbax checkpoint.
- 提供单主机 data parallel strategy 的 CPU correctness 验证.
- 提供模型侧 artifact export 和性能 benchmark report.
- 明确多平台实测,多 GPU 实测,多主机,Grain 和跨拓扑 sharded
  checkpoint 为后续能力.
