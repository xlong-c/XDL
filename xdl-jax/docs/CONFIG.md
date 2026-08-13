# YAML 配置

配置入口是 `xdl_jax.config.setup_from_yaml()`. 根字段使用 schema version 1:

```yaml
config_version: 1
backend: jax
runtime:
  seed: 42
  platform: cpu
trainer:
  max_epochs: 5
  gradient_accumulation_steps: 1
  precision: "32"
task:
  target: xdl_jax.examples:LinearRegressionTask
  params:
    input_dim: 4
train_data:
  target: xdl_jax.examples:SyntheticRegressionData
  params:
    n_samples: 128
    input_dim: 4
    batch_size: 16
optimization:
  optimizer:
    target: optax:adamw
    params:
      learning_rate: 0.05
checkpoint:
  directory: outputs/checkpoints
  every_n_epochs: 1
  max_to_keep: 3
  asynchronous: true
```

`checkpoint` 非空时会自动生成 `CheckpointCallback`. 支持的字段是
`enabled`, `directory`, `every_n_epochs`, `max_to_keep` 和 `asynchronous`.
未知字段直接报错.

代码入口:

```python
from xdl_jax import JaxTrainer
from xdl_jax.config import setup_from_yaml

setup = setup_from_yaml("examples/linear_regression.yaml")
result = JaxTrainer.from_setup(setup).fit_from_setup(setup)
```

