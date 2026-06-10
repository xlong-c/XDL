# XDL 项目结构与使用说明

本文档只回答五件事：

1. XDL 是什么。
2. XDL 的核心层次怎样分工。
3. 当前推荐怎样接入训练。
4. 安装后怎样快速找到用法和公共 API。
5. 新功能应该扩展到哪一层。

## 1. 项目定位

XDL 是一个基于 PyTorch 的模块化深度学习框架。它不替代 PyTorch 的张量、模块和优化器接口，而是在其上补一层更稳定的项目组织能力：

- 组件注册
- YAML 配置构建
- 训练生命周期
- callback 扩展

它更像“项目骨架 + 训练组织层”，而不是新的张量计算框架。

## 2. 核心分层

XDL 的主干可以概括为五层：

```text
组件实现层
  xdl/model xdl/dataset xdl/loss xdl/metric xdl/optimizer xdl/scheduler

组件发现层
  xdl/utils/registry.py

配置构建层
  xdl/config/schema.py / resolver.py / builder.py / setup.py

训练编排层
  xdl/trainer/coreModel.py / trainer.py / trainSetupModel.py

横切扩展层
  xdl/callbacks/
```

对应关系如下：

- `xdl/model/`：模型与工厂函数
- `xdl/dataset/`：数据集、transform、collate
- `xdl/loss/`：损失函数
- `xdl/metric/`：评估指标
- `xdl/optimizer/`：优化器
- `xdl/scheduler/`：学习率调度器
- `xdl/utils/registry.py`：注册系统
- `xdl/config/`：YAML 到 `TrainSetup`
- `xdl/trainer/`：`CoreModel`、`Trainer`、`TrainSetupModel`
- `xdl/callbacks/`：日志、检查点、进度条、早停等横切逻辑

## 3. 三个关键抽象

### `Registry`

`xdl/utils/registry.py` 只负责名字到对象的映射。

它不负责：

- 解析 YAML
- 猜测参数
- 拼装训练流程

这一层越简单，越适合做框架稳定基础设施。

### `CoreModel`

`CoreModel` 承载任务逻辑。新代码推荐从子包入口导入：

```python
from xdl.trainer import CoreModel
```

历史路径 `from xdl.trainer.coreModel import CoreModel` 继续兼容。用户通常需要在子类里实现：

- `training_step()`
- `validation_step()`
- `configure_optimizers()`

XDL 当前采用手动优化模式。也就是说，训练步里需要自行处理：

- `optimizer.zero_grad()`
- `self.manual_backward(loss)`
- `self.clip_gradients(...)`
- `optimizer.step()`
- `self.log("loss", loss, prefix="train")`

简单场景可以使用 `self.manual_optimization_step(loss, ...)` 执行
`zero_grad -> backward(loss / accumulation_steps) -> clip -> step` 模板.

`CoreModel` 会维护训练步计数。新代码优先使用公开属性而不是私有字段：

- `micro_step`：全局 micro-batch 步数，进入 `training_step()` 前已递增。
- `accumulation_steps`：当前梯度累积窗口大小，来自 `Trainer(gradient_accumulation_steps=...)` 或模型侧兼容字段。
- `micro_step_in_accumulation`：当前累积窗口内的 1-based 位置。
- `optimizer_step`：按完整累积窗口推导出的优化器更新次数。
- `is_accumulation_start` / `is_accumulation_boundary` / `should_optimizer_step`：手动累积时判断 `zero_grad()` 与 `optimizer.step()` 的 helper。

日志命名保持旧风格兼容：

- `self.log("train_loss", value)` 继续记录 `train_loss`。
- `self.log("loss", value, prefix="train")` 也记录为 `train_loss`。
- `self.log_metrics({"loss": value}, prefix="val")` 记录为 `val_loss`。
- `value` 支持 Python 数值或单元素 `torch.Tensor`，内部记录为 `float`。
- 已经带 `train_` 或 `train/` 前缀的键不会被重复加前缀。

### `Trainer`

`xdl/trainer/trainer.py` 负责训练循环编排：

- 设备设置
- epoch / step 循环
- callback 调度
- 验证周期
- 推理采样周期

`Trainer.fit()` 在进入训练循环前会先调用 `model.setup("fit")`。重型模块、外部 pipeline、LoRA 之类惰性初始化逻辑应优先放到这里。

训练、验证、测试 batch 会递归迁移常见容器里的 tensor，支持 `Tensor / dict / list / tuple / dataclass`。第三方自定义对象仍应在 `training_step()` / `validation_step()` 中显式处理设备。

外部 pipeline 或非 `nn.Module` 重组件可通过 `CoreModel.configure_device_objects()` 声明给 Trainer 迁移, 并在 `on_after_device_setup()` 中做任务侧收尾。

## 4. 两条推荐使用路径

### 4.1 纯代码路径

适合快速研究和高度定制任务。

入口可以参考：

- [train_VAE.py](../train_VAE.py)
- [train_TwinFlow.py](../train_TwinFlow.py)

典型写法：

```python
from xdl.trainer import CoreModel, Trainer

model = MyCoreModel()
trainer = Trainer(max_epochs=10, device="cuda")
trainer.fit(model, train_loader, val_loader)
```

适用场景：

- 任务逻辑还在快速变化
- 训练步需要高度手写
- 还不值得沉到 YAML

### 4.2 YAML 配置路径

适合标准实验和组件替换。

典型入口：

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

这里的职责分工是：

- `setup_from_yaml()`：解析配置并构建组件
- `TrainSetup`：承载 `model / optimizer / dataloader / loss / metrics`
- `setup.create_model()`：把外部组件包装成 `TrainSetupModel`
- `Trainer.from_setup()`：按配置补齐常用回调

## 5. 安装后单文件入口和公共 API

只安装 wheel、没有源码仓库时，可以直接查看包内用法摘要：

```bash
python -m xdl.usage
xdl-usage
```

Python 内也可以读取同一份文本：

```python
import xdl

print(xdl.get_usage_text())
```

新代码优先依赖这些稳定入口：

```python
from xdl.config import setup_from_yaml, TrainSetup
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
from xdl.callbacks import Callback
```

完整 Stable / Provisional / Internal API 边界见 [API.md](API.md)。

## 6. 当前推荐的接入顺序

当你要接一个新训练任务时，优先按下面顺序阅读：

1. [train_VAE.py](../train_VAE.py)
2. [train_TwinFlow.py](../train_TwinFlow.py)
3. [xdl/trainer/trainer.py](../xdl/trainer/trainer.py)
4. [xdl/trainer/coreModel.py](../xdl/trainer/coreModel.py)
5. [CONFIG.md](CONFIG.md)
6. [API.md](API.md)

这样能最快看清真实生命周期，而不是只看目录名猜结构。

## 7. 怎样扩展 XDL

### 新增模型 / 数据集 / loss / metric / optimizer / scheduler

规则统一：

1. 在对应子模块中实现类或工厂函数
2. 在对应 `__init__.py` 中集中注册
3. 通过 `register_*("Name")(Class)` 接入
4. 在 `__all__` 中导出

不要在定义处直接用装饰器式注册；本仓约定是集中注册。

### 新增 callback

放到 `xdl/callbacks/`，继承 `Callback`，处理自己的生命周期钩子。回调优先作为观察者存在，不要把主要训练逻辑塞进去。

### 新增配置能力

优先判断改动属于哪一层：

- 顶层结构变化：改 `schema.py`
- 配置解析变化：改 `resolver.py`
- 组件实例化变化：改 `builder.py`
- 训练装配变化：改 `setup.py`

## 8. 当前边界

XDL 现在已经具备稳定的“组件注册 + 配置构建 + 训练编排”主链路，但仍有明确边界：

- 不是统一 CLI 框架
- 不是完整实验平台
- 不是自动优化框架
- 分布式和外部大模型接入仍偏工程化

这也是为什么文档和代码都强调“看真实训练入口”，不要假设存在一条包办一切的自动主流程。
