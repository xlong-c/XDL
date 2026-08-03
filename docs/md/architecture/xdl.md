# XDL 架构

本文定义 `XDL` 的架构边界, 核心分层, 生命周期和模块职责. 它是 `XDL` 架构层的长期事实源之一.

## 负责什么

- 说明 `XDL` 是什么, 不是什么.
- 说明核心分层和关键抽象.
- 说明推荐接入路径和扩展落点.
- 说明主要模块的职责边界.

## 不负责什么

- 不重复完整安装命令.
- 不承载长篇操作手册.
- 不单独列出完整 API 兼容清单.

## 项目定位

`XDL` 是一个基于 PyTorch 的模块化深度学习框架. 它不替代 PyTorch 的张量, 模块和优化器接口, 而是在其上补一层更稳定的项目组织能力:

- 组件注册
- YAML 配置构建
- 训练生命周期
- callback 扩展

它更像"项目骨架 + 训练组织层", 而不是新的张量计算框架.

## 核心分层

`XDL` 的主干可以概括为五层:

```text
组件实现层
  xdl/model xdl/dataset xdl/loss xdl/metric xdl/optimizer xdl/scheduler

组件发现层
  xdl/utils/registry.py

配置构建层
  xdl/config/schema.py / resolver.py / builder.py / setup.py

训练编排层
  xdl/trainer/core_model.py / trainer.py / train_setup_model.py

横切扩展层
  xdl/callbacks/
```

对应关系如下:

- `xdl/model/`: 模型与工厂函数
- `xdl/dataset/`: 数据集, transform, collate
- `xdl/loss/`: 损失函数
- `xdl/metric/`: 评估指标
- `xdl/optimizer/`: 优化器
- `xdl/scheduler/`: 学习率调度器
- `xdl/utils/registry.py`: 注册系统
- `xdl/config/`: YAML 到 `TrainSetup`
- `xdl/trainer/`: `CoreModel`, `Trainer`, `TrainSetupModel`
- `xdl/callbacks/`: 日志, 检查点, 进度条, 早停等横切逻辑

## 关键抽象

### `Registry`

`xdl/utils/registry.py` 只负责名字到对象的映射.

它不负责:

- 解析 YAML
- 猜测参数
- 拼装训练流程

这一层越简单, 越适合做框架稳定基础设施.

### `CoreModel`

`CoreModel` 承载任务逻辑. 新代码推荐从子包入口导入:

```python
from xdl.trainer import CoreModel
```

历史路径 `from xdl.trainer.core_model import CoreModel` 继续兼容. 用户通常需要在子类里实现:

- `training_step()`
- `validation_step()`
- `configure_optimizers()`

`XDL` 当前采用手动优化模式. 也就是说, 训练步里需要自行处理:

- `optimizer.zero_grad()`
- `self.manual_backward(loss)`
- `self.clip_gradients(...)`
- `optimizer.step()`
- `self.log("loss", loss, prefix="train")`

简单场景可以使用 `self.manual_optimization_step(loss, ...)` 执行
`zero_grad -> backward(loss / accumulation_steps) -> clip -> step` 模板.

`CoreModel` 会维护训练步计数. 新代码优先使用公开属性而不是私有字段:

- `micro_step`
- `accumulation_steps`
- `micro_step_in_accumulation`
- `optimizer_step`
- `is_accumulation_start`
- `is_accumulation_boundary`
- `should_optimizer_step`

### `Trainer`

`xdl/trainer/trainer.py` 负责训练循环编排:

- 设备设置
- epoch / step 循环
- callback 调度
- 验证周期
- 推理采样周期

`Trainer.fit()` 在进入训练循环前会先调用 `model.setup("fit")`. 重型模块, 外部 pipeline, LoRA 之类惰性初始化逻辑应优先放到这里.

训练, 验证, 测试 batch 会递归迁移常见容器里的 tensor, 支持 `Tensor / dict / list / tuple / dataclass`. 第三方自定义对象仍应在 `training_step()` / `validation_step()` 中显式处理设备.

外部 pipeline 或非 `nn.Module` 重组件可通过 `CoreModel.configure_device_objects()` 声明给 Trainer 迁移, 并在 `on_after_device_setup()` 中做任务侧收尾.

## 推荐使用路径

### 纯代码路径

适合快速研究和高度定制任务.

入口可以参考:

- [train_VAE.py](../../../train_VAE.py)
- [train_TwinFlow.py](../../../train_TwinFlow.py)

典型写法:

```python
from xdl.trainer import CoreModel, Trainer

model = MyCoreModel()
trainer = Trainer(max_epochs=10, device="cuda")
trainer.fit(model, train_loader, val_loader)
```

### YAML 配置路径

适合标准实验和组件替换.

典型入口:

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

这里的职责分工是:

- `setup_from_yaml()`: 解析配置并构建组件
- `TrainSetup`: 承载 `model / optimizer / dataloader / loss / metrics`
- `setup.create_model()`: 把外部组件包装成 `TrainSetupModel`
- `Trainer.from_setup()`: 按配置补齐常用回调

## 模块功能边界

### `xdl/callbacks`

职责:

- 日志, 进度条, 检查点, 早停, 监控等横切逻辑
- 跟随训练生命周期执行

不负责:

- 具体模型训练逻辑
- 优化器步进
- 数据加载

### `xdl/config`

职责:

- 解析 YAML
- 套用 schema v1
- 构建 model / dataset / dataloader / optimizer / scheduler / loss / metrics
- 返回 `TrainSetup`

不负责:

- 替代 `Trainer.fit()`
- 承担模型特化训练逻辑

### `xdl/dataset`

职责:

- 数据集定义
- transform / collate 相关能力
- 通过 registry 对外暴露

不负责:

- 训练循环
- 模型前向

### `xdl/trainer`

职责:

- `CoreModel`: 任务逻辑抽象
- `Trainer`: 训练循环编排
- `TrainSetupModel`: 把配置流构建的外部组件桥接到 `CoreModel`

关键点:

- 稳定公共入口是 `from xdl.trainer import CoreModel, Trainer, TrainSetupModel`
- `Trainer.fit()` 先调用 `model.setup("fit")`
- `CoreModel.training_step()` 是手动优化模式
- 手动累积优先用 `micro_step` / `is_accumulation_boundary` 等公开 helper
- callback 在这里被统一调度

### `xdl/utils`

职责:

- registry
- checkpoint
- tiling
- 权重与通用辅助函数

关键点:

- 这是基础设施层, 改动影响面大
- registry 只做名字到对象的映射, 不掺配置解析

## 依赖方向

可以把依赖方向简化理解成:

```text
model / dataset / loss / metric / optimizer / scheduler
        ↓
      utils.registry
        ↓
      config.builder / setup
        ↓
       trainer
        ↓
     callbacks
```

更准确地说:

- 组件层依赖 registry 暴露自己
- config 依赖 registry 构建组件
- trainer 依赖 config 产物或纯代码组件
- callbacks 依赖 trainer 生命周期

## 扩展点

- 新增模型 / 数据集 / loss / metric / optimizer / scheduler: 在对应子模块实现, 在对应 `__init__.py` 中集中注册.
- 新增 callback: 放到 `xdl/callbacks/`, 继承 `Callback`, 处理自己的生命周期钩子.
- 新增配置能力: 按层次修改 `schema.py`, `resolver.py`, `builder.py`, `setup.py`.

## 禁止项

- 不要在定义处直接用装饰器式注册; 本仓约定是集中注册.
- 不要把主要训练逻辑塞进 callback.
- 不要假设存在一条包办一切的自动主流程.

## 继续阅读

- [api-boundary.md](api-boundary.md)
- [../usage/xdl-install-and-verify.md](../usage/xdl-install-and-verify.md)
- [../usage/index.md](../usage/index.md)
