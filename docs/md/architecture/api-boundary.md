# XDL API 稳定边界

本文定义 `XDL` 对外承诺的 Python API 边界. 它不替代使用教程, 也不列出每个模型或损失函数的参数.

## 负责什么

- 定义 Stable / Provisional / Internal 三类 API.
- 说明兼容路径, 废弃策略和版本规则.

## 不负责什么

- 不提供完整使用教程.
- 不展开每个组件的参数细节.

## 稳定级别

`XDL` API 分为三类:

- Stable: 推荐用户和训练脚本直接依赖. 兼容性变更需要保留旧路径并给迁移说明.
- Provisional: 可以使用, 但仍可能随训练能力和配置系统演进调整.
- Internal: 内部实现细节, 不承诺兼容. 用户代码不要直接依赖.

当前项目版本仍是 `0.x`, Stable 表示"本仓库内优先保持兼容", 不是已经完成 `1.0` 级别冻结.

## Stable API

### 配置入口

```python
from xdl.config import setup_from_yaml, TrainSetup, load_structured_dataclass_config
```

承诺:

- `setup_from_yaml(config_path, device=None)` 是 YAML 配置主入口.
- 返回值是 `TrainSetup`.
- `TrainSetup.create_model()` 返回可交给 `Trainer.fit()` 的 `CoreModel` 包装对象.

### 训练入口

```python
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
```

承诺:

- `Trainer` 是训练主循环入口.
- `Trainer.fit(model, train_dataloader, val_dataloader=None, inference_data=None)` 是训练主路径.
- `CoreModel` 是用户自定义任务逻辑的基类.
- `TrainSetupModel` 是配置流到 `CoreModel` 的适配层.

兼容路径:

```python
from xdl.trainer.coreModel import CoreModel
from xdl.trainer.trainer import Trainer
from xdl.trainer.trainSetupModel import TrainSetupModel
```

### 回调入口

```python
from xdl.callbacks import Callback
```

承诺:

- `Callback` 是自定义训练生命周期扩展的基类.
- 回调优先级规则保持稳定: 数值越小越先执行.

### Registry 入口

```python
from xdl.utils.registry import (
    Registry,
    register_model,
    register_dataset,
    register_optimizer,
    register_scheduler,
    register_loss,
    register_metric,
    register_transform,
    register_collate,
)
```

承诺:

- `Registry` 提供注册, 查找, 列举能力.
- `register_*("Name")(ClassOrFunction)` 是自定义组件接入方式.
- 支持的注册类型保持为: `MODEL`, `DATASET`, `OPTIMIZER`, `SCHEDULER`, `LOSS`, `METRIC`, `TRANSFORM`, `COLLATE`.

### 常用工具入口

```python
from xdl.utils import resolve_dtype, save_yaml, seed_everything
```

### 安装后用法入口

```python
import xdl

text = xdl.get_usage_text()
xdl.print_usage()
```

命令行入口:

```bash
python -m xdl.usage
xdl-usage
```

## Provisional API

以下 API 当前可用, 但仍处于演进期:

- `xqt` 顶层实验入口及其结果对象.
- `xqt` 到 `XDL` 的适配入口.
- `xdl.config` 中的 schema dataclass 与 resolver 工具.
- Accelerate, DeepSpeed, FSDP 相关配置字段和行为.
- 具体内置模型, 数据集, loss, metric, optimizer, scheduler 的注册名称和参数细节.

使用这些 API 时, 建议通过测试固定自己的项目契约.

## Internal API

以下内容不承诺兼容:

- 以下划线开头的函数, 类, 属性和模块级变量.
- `xdl.config.builder`, `xdl.config.setup` 中的内部辅助函数.
- `Trainer`, `CoreModel`, callback 内部状态字段, 除公开 property 和文档明确说明的字段外.
- 临时研究, 实验, 工具目录中的脚本入口.

## 废弃策略

对 Stable API 做破坏性变更时, 遵守下面流程:

1. 新增推荐路径或新参数.
2. 旧路径继续保留, 并在必要时发出 `DeprecationWarning`.
3. 文档写清楚替代方式.
4. 至少保留一个后续版本窗口.
5. 删除旧 API 只能发生在明确的破坏性版本中.

## 版本规则

建议按语义化版本管理:

- Patch: bugfix, 文档, 测试, 非破坏性兼容修复.
- Minor: 新增公共能力, 扩展配置字段, 增加组件.
- Major: 删除或破坏 Stable API.

当前 `0.x` 阶段仍允许调整设计, 但应避免无提示破坏本文档列出的 Stable API.
