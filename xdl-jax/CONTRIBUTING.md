# xdl-jax 开发指南

## 环境

项目要求 Python `>=3.12`.

CPU 开发环境:

```bash
python -m pip install -e '.[cpu,dev]'
```

CUDA 13 开发环境:

```bash
python -m pip install -e '.[gpu-cuda13,dev]'
```

## 本地检查

在仓库根目录执行:

```bash
XDL_PUNCT_CHECK=1 XDL_PUNCT_PATHS=xdl-jax python scripts/normalize_punctuation.py
ruff check xdl-jax/xdl_jax xdl-jax/tests xdl-jax/examples
pyright xdl-jax/xdl_jax
JAX_PLATFORMS=cpu PYTHONPATH=xdl-jax python -m pytest xdl-jax/tests -q
```

GPU smoke 需要串行执行,并关闭 JAX 默认显存预分配:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=xdl-jax \
  python -m pytest xdl-jax/tests/test_gpu.py -q
```

## 代码约束

- 所有函数加类型注解.
- 训练状态保持 immutable PyTree.
- 参数, mutable state, optimizer state, RNG 和 loop counters 不混用.
- 配置使用 OmegaConf 和 `target + params`,不做隐式字段迁移.
- 新增公开 API 时同步更新 `docs/API.md` 和 `DESIGN.md`.
- 不在 `xdl-jax` 引入 Torch 训练类型,不把训练职责放入 XQT.
- 多平台和多 GPU 没有运行证据时,保持明确的限制描述.

## 版本和发布

版本来源是 `xdl_jax/_version.py`. 版本变更同步:

- `CHANGELOG.md`.
- `pyproject.toml` 的发布元数据.
- 对应的测试和运行证据.

构建 wheel:

```bash
python -m build
```

