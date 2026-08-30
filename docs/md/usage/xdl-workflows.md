# XDL 工作流

本文汇总 `XDL` 当前推荐的工作流入口. 它聚焦"怎么做", 不重复完整架构背景.

## 负责什么

- 说明纯代码路径和 YAML 路径的最短工作流.
- 说明新增组件时的接入顺序.
- 指向相关架构和 API 文档.

## 不负责什么

- 不重复环境安装步骤.
- 不替代完整 API 边界文档.
- 不展开所有配置字段细节.

## 纯代码路径

适合快速研究和高度定制任务.

推荐先看:

1. [../../../train_VAE.py](../../../train/pretrain/train_VAE.py)
2. [../../../train_TwinFlow.py](../../../train/pretrain/train_TwinFlow.py)
3. [../../../xdl/trainer/trainer.py](../../../xdl/trainer/trainer.py)
4. [../../../xdl/trainer/core_model.py](../../../xdl/trainer/core_model.py)

典型写法:

```python
from xdl.trainer import CoreModel, Trainer

model = MyCoreModel()
trainer = Trainer(max_epochs=10, device="cuda")
trainer.fit(model, train_loader, val_loader)
```

适用场景:

- 任务逻辑还在快速变化
- 训练步需要高度手写
- 还不值得沉到 YAML

## 大模型多卡训练要点

10B 级扩散模型 / LoRA 微调的多卡场景, 先对齐这几个入口:

- FSDP: `Trainer(fsdp=...)` 支持 int / dict / `FSDPConfig`; transformer 按
  block wrap 用 `transformer_cls_names_to_wrap`, 冻结组件
  (VAE / text encoder) 在 `CoreModel.configure_unwrapped_modules()` 里跳过
  `accelerator.prepare`.
- checkpoint: FSDP 下默认 `pt` 格式会自动聚合模型权重; 需要完整恢复
  优化器/调度器时用 `format="accelerator"`.
- 采样: 模型声明 `requires_collective_sampling` 后, 所有 rank 一起前向,
  主 rank 落盘; 采样回调开 `collective=True`.
- 数值稳定性: `grad_clip_max_norm` 已生效; `nan_monitor` 默认开启, 连续
  NaN 步自动中止.
- 显存: `GradientCheckpointingCallback` + `ActivationOffloadCallback`
  (选择性激活卸载); 条件预编码缓存由 `CoreModel` 在
  `setup()` / `on_after_device_setup()` 里自行存/自行加载.

完整说明见 [xdl/trainer/README.md](../../../xdl/trainer/README.md).

## YAML 配置路径

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

职责分工:

- `setup_from_yaml()`: 解析配置并构建组件
- `TrainSetup`: 承载 `model / optimizer / dataloader / loss / metrics`
- `setup.create_model()`: 把外部组件包装成 `TrainSetupModel`
- `Trainer.from_setup()`: 按配置补齐常用回调

## 新增能力时的接入顺序

当你要接一个新训练任务时, 优先按下面顺序阅读:

1. [../../../train_VAE.py](../../../train/pretrain/train_VAE.py)
2. [../../../train_TwinFlow.py](../../../train/pretrain/train_TwinFlow.py)
3. [../../../xdl/trainer/trainer.py](../../../xdl/trainer/trainer.py)
4. [../../../xdl/trainer/core_model.py](../../../xdl/trainer/core_model.py)
5. [../README.md#xdl-config-系统说明](../README.md#xdl-config-系统说明)
6. [../architecture/api-boundary.md](../architecture/api-boundary.md)

这样能最快看清真实生命周期, 而不是只看目录名猜结构.

## 扩展方式

### 新增模型 / 数据集 / loss / metric / optimizer / scheduler

规则统一:

1. 在对应子模块中实现类或工厂函数
2. 在对应 `__init__.py` 中集中注册
3. 通过 `register_*("Name")(Class)` 接入
4. 在 `__all__` 中导出

### 新增 callback

放到 `xdl/callbacks/`, 继承 `Callback`, 处理自己的生命周期钩子. 回调优先作为观察者存在, 不要把主要训练逻辑塞进去.

### 新增配置能力

优先判断改动属于哪一层:

- 顶层结构变化: 改 `schema.py`
- 配置解析变化: 改 `resolver.py`
- 组件实例化变化: 改 `builder.py`
- 训练装配变化: 改 `setup.py`

## 继续阅读

- [xdl-install-and-verify.md](xdl-install-and-verify.md)
- [xdl-agent-handoff.md](xdl-agent-handoff.md)
- [../architecture/xdl.md](../architecture/xdl.md)
