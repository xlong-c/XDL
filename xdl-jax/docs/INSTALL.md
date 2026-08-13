# 安装和验证

## CPU

```bash
python -m pip install -e '.[cpu,test]'
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python examples/run_linear_regression.py
```

## NVIDIA GPU

CUDA 12:

```bash
python -m pip install -e '.[gpu-cuda12,test]'
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
  python -m pytest tests/test_gpu.py -q
```

CUDA 13:

```bash
python -m pip install -e '.[gpu-cuda13,test]'
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
  python -m pytest tests/test_gpu.py -q
```

GPU 运行时必须显式选择 `platform: gpu` 或
`TrainerConfig(platform="gpu")`. 缺少 CUDA-enabled JAX plugin 时会抛出
明确错误,不会静默回退到 CPU.

## 依赖约束

- `constraints/cpu.txt`: CPU tested range.
- `constraints/cuda12.txt`: CUDA 12 optional range.
- `constraints/cuda13.txt`: CUDA 13 optional range.

