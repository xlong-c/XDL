# XDL 配置工作流

本文承接 `XDL` 配置系统的使用层正文. 它聚焦"怎么通过配置组织训练组件", 不重复完整架构背景.

## 负责什么

- 说明 `setup_from_yaml()` 主链路.
- 说明 `target + params` 的基本写法.
- 说明 dataset / dataloader / collate 在配置中的关系.

## 不负责什么

- 不替代架构层对模块边界的定义.
- 不展开完整 dataset 模板规划.
- 不列出所有 schema dataclass 细节.

## 主入口

当前推荐入口:

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
```

主链路理解为:

```text
YAML
  -> schema / resolver
  -> builder
  -> TrainSetup
  -> TrainSetupModel / Trainer.from_setup()
```

## `target + params`

配置主链路遵循:

- 优先 structured config / dataclass schema
- YAML 组织方式以 `target + params` 为主
- 加载, 合并, 插值和容器转换优先交给 `OmegaConf`

不要在训练入口散落手写 `yaml.safe_load` 合并逻辑.

## dataset / dataloader / collate

配置侧要区分三件事:

- dataset: 样本如何定义
- dataloader: batch 如何产生
- collate: batch 字段如何拼接

基本规则:

- dataset 本身使用 `target + params`
- dataloader 通过 `dataset: ${train_dataset}` 复用 dataset 配置
- `collate_fn` 需要特殊处理时也使用 `target + params`

## 典型入口

如果要走 YAML 配置路径, 当前推荐直接在 Python 中调用:

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

## 相关阅读

- [../architecture/xdl.md](../architecture/xdl.md)
- [xdl-workflows.md](xdl-workflows.md)
- [../README.md#xdl-dataset-模板规划](../README.md#xdl-dataset-模板规划)
