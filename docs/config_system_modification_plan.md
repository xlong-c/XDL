# XDL Config 系统修改计划与预期效果

编写时间: 2026-03-18

## 文档目的

本文档用于明确 XDL config 系统后续应如何改造，重点回答两个问题：

1. 需要做哪些修改。
2. 每一类修改完成后会带来什么效果。

本文档是实施计划，不是现状评估。当前问题清单和证据请参考：

- [config_system_assessment.md](/root/workspace/xdl/docs/config_system_assessment.md)

## 总体目标

XDL config 系统的改造目标不是简单“把 YAML 读出来”，而是形成一套可维护、可验证、可扩展的统一配置体系。

最终目标分为两层：

- **V1：稳定的组件构建系统**
  - 能可靠构建 `model`、`dataset`、`dataloader`、`optimizer`、`scheduler`、`loss`、`metrics`
  - 能返回可靠的 `TrainSetup`
  - 能对错误配置做明确失败

- **V2：完整的实验编排系统**
  - 在 V1 基础上接入 `Trainer`
  - 接入 logger、checkpoint、callbacks、Accelerate
  - 让 YAML 从“组件配置”升级为“实验配置”

## 设计原则

改造过程中，建议统一遵守以下原则：

1. **Schema-first**
   - 先定义配置结构，再写 builder。
   - 不允许长期存在“示例里支持，但代码里不消费”的字段。

2. **Registry-first**
   - XDL 本地组件优先通过 registry 查找。
   - 对外部库组件使用清晰的 import 路径，而不是模糊的库别名推断。

3. **Explicit-failure**
   - 未注册组件、错误字段、未消费字段、无效引用必须明确失败或明确告警。
   - 不再接受静默跳过。

4. **Single-schema**
   - 所有官方 YAML 必须使用同一套顶层结构、字段命名和组件描述格式。
   - 不再允许 `num_epochs` / `max_epochs`、`grad_steps` / `gradient_accumulation_steps` 这种长期并存。

5. **Dataclass-locked upper structure**
   - 能合并、能统一、跨实验高复用的上层结构，优先用 `dataclass` 固定。
   - 允许灵活变化的部分尽量下沉到组件级 `params`，而不是放在顶层结构漂移。
   - 顶层结构的新增字段必须先进入 dataclass schema，再进入官方 YAML。

6. **Progressive integration**
   - 先修好组件构建主链路，再接训练编排层。
   - 不建议在基础层不稳定时直接引入完整 Hydra 范式。

## 推荐目标结构

后续官方配置建议统一到下面的顶层结构：

```yaml
config_version: 1

runtime:
  device: cuda
  seed: 42
  data_dir: ./data
  output_dir: ./others
  experiment_name: demo

trainer:
  max_epochs: 100
  batch_size: 128
  precision: "32"
  gradient_accumulation_steps: 1
  grad_clip_max_norm: 5.0

model:
  target: registry:vgg16_bn
  params:
    num_classes: 100
    dropout: 0.5

dataloader_defaults:
  batch_size: ${trainer.batch_size}
  num_workers: 4
  pin_memory: true

train_transforms: ...
val_transforms: ...
train_dataset: ...
val_dataset: ...
train_dataloader: ...
val_dataloader: ...

optimization:
  optimizer: ...
  scheduler: ...

loss: ...

metrics: ...

logging:
  ...

checkpoint:
  ...

accelerate:
  ...
```

统一后的配置风格建议：

- 所有组件使用同一描述格式：`target + params`
- 所有训练行为放到 `trainer`
- 所有运行环境放到 `runtime`
- 数据链路优先使用顶层紧凑别名，减少一层 `data` 缩进
- 所有日志与保存分离成 `logging`、`checkpoint`

## Dataclass 固定策略

建议把“上层结构”与“底层组件参数”明确分开：

- **上层结构**：使用 `dataclass` 固定
  - 例如 `runtime`、`trainer`、`data`、`optimization`、`logging`、`checkpoint`、`accelerate`
- **底层组件参数**：保留在 `params: Dict[str, Any]`
  - 例如 `model.params`
  - 例如 `optimizer.params`
  - 例如 `scheduler.params`
  - 例如第三方 transform / dataset / metric 的构造参数

这样做的核心原因是：

- 上层结构是跨项目、跨实验高频复用的“稳定骨架”，应该固定。
- 底层组件参数随模型、数据集、库版本变化更频繁，应该保留灵活性。

推荐 dataclass 分层如下：

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./others"
    experiment_name: str = "default_exp"


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    precision: str = "32"
    gradient_accumulation_steps: int = 1
    grad_clip_max_norm: Optional[float] = None


@dataclass
class ComponentConfig:
    type: str
    source: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    transforms: Dict[str, ComponentConfig] = field(default_factory=dict)
    datasets: Dict[str, ComponentConfig] = field(default_factory=dict)
    dataloaders: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    optimizer: ComponentConfig = field(default_factory=lambda: ComponentConfig(target=""))
    scheduler: Optional[ComponentConfig] = None


@dataclass
class ConfigSchemaV1:
    config_version: int = 1
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ComponentConfig = field(default_factory=lambda: ComponentConfig(target=""))
    data: DataConfig = field(default_factory=DataConfig)
    optimization: Optional[OptimizationConfig] = None
    loss: Optional[ComponentConfig] = None
    metrics: List[ComponentConfig] = field(default_factory=list)
```

这个策略的边界建议明确如下：

- 顶层字段是否存在，由 dataclass 决定。
- 顶层字段的类型，由 dataclass 决定。
- 顶层字段的默认值，由 dataclass 决定。
- 组件内部具体参数是否有效，由 builder 和组件构造器决定。

预期效果：

- 顶层结构不会因为新增示例而不断漂移。
- YAML 合并和默认值策略更稳定。
- `OmegaConf` 的 structured config 能真正发挥作用。
- 文档、代码、测试能围绕同一份 schema 收敛。

## 修改计划

### Phase 0：修复当前主链路阻断问题

本阶段目标是让当前 config 系统“至少能正常导入、能跑通一个最小示例”。

#### 计划修改的模块

- [dataclass.py](/root/workspace/xdl/xdl/config/dataclass.py)
- [setup.py](/root/workspace/xdl/xdl/config/setup.py)
- [builder.py](/root/workspace/xdl/xdl/config/builder.py)
- [vgg_cifar100.yaml](/root/workspace/xdl/config/vgg_cifar100.yaml)
- [unified_logger_example.yaml](/root/workspace/xdl/config/unified_logger_example.yaml)

#### 具体改动

1. 修复 `TrainSetup` 的 dataclass 字段顺序问题。
2. 修复 `_get_class_from_library` 对 `torch` / `torchvision` 的错误映射。
3. 删除 `build_model` 中对 `num_classes -> kwargs` 的魔法改写。
4. 修复 `build_transform` 中无效的 `sequential` 分支，或移除未实现分支。
5. 让最小官方示例 YAML 真正能构建成功。
6. 对示例中的明显无效配置项做清理或标注，例如 `simple_mlp`、`CustomDataset` 未实现时不能继续作为“官方可运行示例”。

#### 预期效果

- `from xdl.config import setup_from_yaml` 可成功执行。
- 至少 1 个官方 YAML 能在 CPU 下构建成功。
- `model`、`optimizer`、`loss`、`metrics` 等主链路组件可被真实实例化。
- 当前“文档可用、运行不可用”的最严重问题被消除。

#### 验收标准

- `setup_from_yaml('config/vgg_cifar100.yaml', device='cpu')` 成功返回 `TrainSetup`
- `TrainSetup.model`、`optimizer`、`loss_fn`、`metrics` 类型正确
- 构建失败时错误信息指向具体组件和字段

### Phase 1：统一 schema、风格和结构

本阶段目标是让 config 体系从“可用”升级为“统一、严格、可维护”。

#### 计划新增/修改的模块

- 新增 [schema.py](/root/workspace/xdl/xdl/config/schema.py)
- 新增 [errors.py](/root/workspace/xdl/xdl/config/errors.py)
- 新增 [resolver.py](/root/workspace/xdl/xdl/config/resolver.py)
- 修改 [setup.py](/root/workspace/xdl/xdl/config/setup.py)
- 修改 [builder.py](/root/workspace/xdl/xdl/config/builder.py)
- 修改 [__init__.py](/root/workspace/xdl/xdl/config/__init__.py)
- 修改官方 YAML 示例

#### 具体改动

1. 设计统一 schema。
   - 固定顶层结构。
   - 统一字段命名。
   - 统一组件描述格式。
   - 用 dataclass 固定高层结构，不让顶层 schema 漫游。

2. 引入结构化配置层。
   - 推荐使用 `dataclass + OmegaConf`。
   - OmegaConf 只负责配置对象、合并、插值、类型约束，不直接替代 XDL builder。
   - `OmegaConf.structured(...)` 生成的配置对象应以顶层 dataclass 为准。

3. 引入统一错误模型。
   - `ConfigError`
   - `ConfigValidationError`
   - `ComponentResolutionError`
   - `UnusedConfigWarning`

4. 对配置做严格校验。
   - 缺少必填字段时报错。
   - 字段类型错误时报错。
   - 未知字段报错或至少强告警。
   - 未消费字段必须显式提示。

5. 统一组件描述方式。
   - 从当前混合的 `name` / `from_library` / `backbone` / `main_optimizer` 收敛到统一格式。
   - 推荐格式：`target + params`

6. 明确“固定层”和“灵活层”边界。
   - `runtime` / `trainer` / `logging` / `checkpoint` / `accelerate` 等固定层使用 dataclass。
   - 组件级构造参数使用 `params` 保持灵活。

#### 预期效果

- 所有官方配置文件长得像同一种语言。
- 用户不再需要记住每个块的特殊写法。
- 配置错误能在 build 前暴露，而不是在训练中后期才暴露。
- `setup_from_yaml` 的职责更清晰，只负责调度，不再负责猜测配置意图。
- 顶层结构可以稳定合并、稳定默认化、稳定迁移。

#### 验收标准

- 所有官方 YAML 通过 schema 校验
- 未知字段会被明确提示
- 同一类组件的配置结构完全一致
- `num_epochs` / `max_epochs` 这类历史歧义被清理

### Phase 2：补齐 registry 和引用解析闭环

本阶段目标是让“本地组件可配置化构建”成为真实能力，而不是接口占位。

#### 计划新增/修改的模块

- 新增 [__init__.py](/root/workspace/xdl/xdl/dataset/__init__.py)
- 修改 [__init__.py](/root/workspace/xdl/xdl/__init__.py)
- 修改 [registry.py](/root/workspace/xdl/xdl/utils/registry.py)
- 修改 [builder.py](/root/workspace/xdl/xdl/config/builder.py)
- 修改 [resolver.py](/root/workspace/xdl/xdl/config/resolver.py)

#### 具体改动

1. 为 `xdl/dataset` 增加统一注册入口。
2. 在顶层 `xdl` 初始化时纳入 dataset 模块导入。
3. 明确 transform 是否支持 registry。
   - 如果支持，就打通 `TRANSFORM_REGISTRY`
   - 如果不支持，就移除相关承诺，避免文档误导
4. 实现配置引用解析。
   - 支持 `${train_transforms}` 与 `${data.transforms.train}`
   - 支持 `${train_dataset}`
   - 支持 `${optimization.optimizer}`
5. 对引用不存在、路径错误、循环引用等情况做明确失败。

#### 预期效果

- 本地 dataset 可以通过 YAML 被发现和实例化。
- transform、dataset、dataloader、optimizer、scheduler 之间的依赖关系可被显式表达。
- 配置文件不再只是“长得像配置图”，而是真正有依赖解析能力。

#### 验收标准

- `DATASET_REGISTRY` 不再为空
- 至少 1 个本地 dataset 可通过 YAML 构建成功
- `${...}` 风格引用可被成功解析
- 引用错误时给出明确报错

### Phase 3：接入 Trainer、logging、checkpoint、Accelerate

本阶段目标是把 config 系统从“组件构建系统”升级为“实验编排系统”。

#### 计划新增/修改的模块

- 修改 [setup.py](/root/workspace/xdl/xdl/config/setup.py)
- 修改 [dataclass.py](/root/workspace/xdl/xdl/config/dataclass.py)
- 修改 [accelerate_config.py](/root/workspace/xdl/xdl/config/accelerate_config.py)
- 新增 `trainer_builder.py`
- 新增 `callback_builder.py`
- 新增 `logger_builder.py`
- 修改相关 callbacks / trainer 接口

#### 具体改动

1. 把 `trainer` 配置真正映射到 `Trainer` 构造参数。
2. 把 `logging` 配置映射到 logger / TensorBoard / console / WandB。
3. 把 `checkpoint` 配置映射到 `ModelCheckpoint` 等 callback。
4. 把 `accelerate` 配置映射到 `AccelerateConfig`。
5. 允许新增高层入口，例如：
   - `setup_from_yaml()` 返回 `TrainSetup`
   - `build_trainer_from_yaml()` 返回完整 `Trainer + setup`

#### 预期效果

- YAML 不再只是拼训练组件，而是能定义完整实验。
- logger、checkpoint、Accelerate 不再是“配置里有，但主流程不接”的悬空字段。
- 框架文档可以真正推荐统一 YAML 工作流。

#### 验收标准

- 一份 YAML 可以一键完成完整训练初始化
- `logging`、`checkpoint`、`accelerate` 字段真实生效
- 训练脚本可以明显减少手动 glue code

### Phase 4：测试、迁移和文档收尾

本阶段目标是让改造结果可维护、可回归、可迁移。

#### 计划新增/修改的模块

- 新增 `tests/config/`
- 新增 `docs/config_schema_v1.md`
- 新增 `docs/config_migration.md`
- 修改 README 与 AGENTS 文档

#### 具体改动

1. 增加 config smoke tests。
2. 为每份官方 YAML 增加至少一个构建测试。
3. 增加无效配置测试。
4. 编写 schema 文档与迁移文档。
5. 清理 README、AGENTS、子模块文档中的历史写法和无效示例。

#### 预期效果

- 配置系统不再依赖“人工记忆”维持稳定。
- 示例 YAML 与代码实现保持同步。
- 后续重构 builder 时不容易打断现有功能。

#### 验收标准

- 官方 YAML 全部有测试覆盖
- 关键错误路径有测试覆盖
- 文档中的示例代码和 YAML 可真实执行

## 推荐实施顺序

建议严格按下面顺序推进：

1. **先做 Phase 0**
   - 先把主链路修通，不要同时做结构大改。

2. **再做 Phase 1**
   - schema、命名和风格统一，是后续所有功能扩展的基础。

3. **然后做 Phase 2**
   - 没有 registry 闭环和引用解析，就不要急着扩展上层配置能力。

4. **最后再做 Phase 3**
   - 训练编排是上层能力，应该建立在稳定组件构建层之上。

5. **全过程都同步推进 Phase 4 的测试和文档**

## OmegaConf 的角色建议

建议在 Phase 1 引入 `OmegaConf`，但只让它负责配置层，不替代 XDL 的 builder。

推荐分工：

- `OmegaConf`
  - 读取 YAML
  - 合并默认配置
  - 处理 `${...}` 插值
  - 做结构化配置和类型约束
  - 以顶层 dataclass schema 固定上级结构

- `XDL builder / registry`
  - 查找组件
  - 解析本地 registry
  - 构建模型、数据、优化器、scheduler 等对象

预期效果：

- 能统一配置风格和结构
- 不需要立刻切换到 Hydra 的完整范式
- 可在后续需要时再增加 Hydra 适配层

## 不建议当前阶段直接做的事情

以下事项不建议在主链路未收敛前优先推进：

1. 直接把整个 config 系统迁移为 Hydra `_target_` 风格。
2. 同时保留两套长期并存的官方 schema。
3. 在没有 schema 校验前继续扩展更多示例 YAML。
4. 在 builder 里继续增加隐式推断和魔法字段修正。

原因很简单：这些做法会继续放大当前“风格不统一、语义不一致、示例不可跑”的问题。

## 最终预期效果

如果按本文档推进，XDL config 系统最终应达到下面这些效果：

1. **可用**
   - config 主入口可导入
   - 官方 YAML 可真实构建成功

2. **统一**
   - 所有配置都遵循同一结构和命名规范
   - 所有组件都使用同一种描述方式

3. **可信**
   - 错误配置能尽早失败
   - 未使用字段不会被静默吞掉
   - 文档与实现一致

4. **可扩展**
   - 新增 model / dataset / metric / loss 时能平滑注册和接入
   - 后续可继续接 callbacks、logger、Accelerate

5. **可维护**
   - 配置系统有测试
   - 示例有 smoke 验证
   - 迁移路径清晰

## 建议的完成定义

可以把 config 系统“完成第一阶段改造”的标准定义为：

- `setup_from_yaml` 稳定可用
- 至少 2 份官方 YAML 能构建成功
- schema 统一
- registry 闭环补齐
- 配置错误有明确报错
- 官方文档全部同步更新

达到这个标准后，再去扩展 Hydra 适配、复杂 sweep 或更大规模实验编排，会更稳妥。
