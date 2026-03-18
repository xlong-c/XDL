# XDL Config Schema V1 落地说明

编写时间: 2026-03-18

本文档用于说明 XDL 当前已经落地的配置系统能力，重点回答四个问题：

1. 现在官方推荐使用什么配置结构。
2. `OmegaConf` 在 XDL 里负责什么，不负责什么。
3. `transform: ${data.transforms.train}` 这类写法到底是怎么工作的。
4. 原先 builder / registry 存在哪些问题，本轮改造修了哪些，哪些还没接完。

如需看问题诊断和演进路线，请分别参考：

- [config_system_assessment.md](/root/workspace/xdl/docs/config_system_assessment.md)
- [config_system_modification_plan.md](/root/workspace/xdl/docs/config_system_modification_plan.md)

## 当前结论

截至本轮改造，XDL config 系统已经从“松散 YAML + 手工猜测字段”收敛为“dataclass 固定上层结构 + OmegaConf 负责配置层 + builder/registry 负责实例化”的模式。

当前已经稳定支持：

- `model`
- `data.transforms`
- `data.datasets`
- `data.dataloaders`
- `optimization.optimizer`
- `optimization.scheduler`
- `loss`
- `metrics`
- `TrainSetup` 返回

当前还没有完全接通到训练编排主链路的能力：

- `logging` 目前进入 schema，但还没有统一映射到 callback / logger builder
- `checkpoint` 目前进入 schema，但还没有统一映射到 checkpoint callback 构建
- `accelerate` 目前进入 schema，但还没有接到完整训练流程

所以目前对 config 系统更准确的定位仍然是：

- **已稳定的组件构建系统**
- **未完全闭环的实验编排系统**

## 官方推荐结构

当前官方 YAML 应统一使用 schema v1：

```yaml
config_version: 1

runtime:
  device: cuda
  seed: 42
  output_dir: ./others
  experiment_name: demo

trainer:
  max_epochs: 100
  batch_size: 128
  precision: "32"
  gradient_accumulation_steps: 1
  grad_clip_max_norm: 5.0

model:
  type: vgg16_bn
  source: registry
  params:
    num_classes: 100

data:
  transforms:
    train: ...
    val: ...
  datasets:
    train: ...
    val: ...
  dataloaders:
    train: ...
    val: ...

optimization:
  optimizer: ...
  scheduler: ...

loss:
  - type: CrossEntropyLoss
    source: torch.nn
    params: {}

metrics:
  - type: Accuracy
    source: registry
    params:
      num_classes: 100

logging:
  ...

checkpoint:
  ...

accelerate:
  ...
```

统一规则如下：

- 顶层结构由 dataclass 固定，不再让官方 YAML 自由漂移
- 组件统一使用 `type + source + params`
- 复杂且高变动的构造参数，继续放在组件级 `params`
- 官方示例不再继续传播 `core_config`、`data_config`、`from_library`、`backbone` 这类旧结构写法

## Dataclass 固定了什么

当前 schema 主要定义在 [schema.py](/root/workspace/xdl/xdl/config/schema.py)。

被固定住的，是“上层结构”和“骨架字段”：

- `runtime`
- `trainer`
- `model`
- `data`
- `optimization`
- `loss`
- `metrics`
- `logging`
- `checkpoint`
- `accelerate`

这意味着：

- 顶层字段有没有，由 dataclass 决定
- 顶层字段类型是否正确，由 dataclass + OmegaConf 决定
- 默认值是什么，由 dataclass 决定

而下面这些内容仍然保持灵活：

- `model.params`
- `optimizer.params`
- `scheduler.params`
- 第三方 transform / dataset / metric 的构造参数

这样做的核心原因是：

- 上层结构应该稳定
- 底层组件参数天然高变动，不适合全部硬编码死

## OmegaConf 在 XDL 里的角色

当前 `OmegaConf` 的职责很明确，只负责“配置层”：

- 从 YAML / dict 加载配置
- 把用户配置 merge 到 dataclass schema
- 做 `${...}` 插值解析
- 输出结构化的 `DictConfig`

当前 `OmegaConf` **不负责**：

- 直接构建 PyTorch 对象
- 替代 XDL 的 registry
- 替代 builder

实际链路如下：

```text
YAML / dict
  -> load_config_with_schema(...)
  -> OmegaConf.structured(schema)
  -> OmegaConf.merge(schema_cfg, raw_cfg)
  -> OmegaConf.resolve(...)
  -> to_plain_dict(...)
  -> build_model / build_dataset / build_optimizer / ...
```

也就是说，XDL 现在是：

- `OmegaConf` 负责“配置对象”
- `builder + registry` 负责“运行对象”

这正是当前阶段最稳妥的边界。

## `${data.transforms.train}` 是怎么实现的

以这段配置为例：

```yaml
data:
  transforms:
    train:
      type: compose
      items:
        - type: ToTensor
          source: torchvision.transforms
          params: {}

  datasets:
    train:
      type: CIFAR100
      source: torchvision.datasets
      params:
        root: ./data
        train: true
      transform: ${data.transforms.train}
```

它不是 Python 运行时直接共享对象，也不是 builder 在字符串里自己做路径查找，而是分两步完成：

1. **配置插值阶段**
   - `load_config_with_schema(..., resolve=True)` 会调用 `OmegaConf.resolve(...)`
   - `${data.transforms.train}` 会被解析成 `data.transforms.train` 对应的那一整段配置对象
   - 这一步完成后，`dataset.transform` 已经不再是字符串，而是一个普通 dict

2. **实例化阶段**
   - `setup_from_yaml(...)` 先构建 `data.transforms`
   - 再读取 dataset 配置里的 `transform`
   - 如果 `transform` 是 dict，就调用 `build_transform(...)`
   - 最终把真实 transform 对象塞进 dataset 构造参数里

因此这类写法的本质是：

- **OmegaConf 负责引用展开**
- **XDL builder 负责对象实例化**

对应实现位置：

- 配置解析： [resolver.py](/root/workspace/xdl/xdl/config/resolver.py)
- 构建调度： [setup.py](/root/workspace/xdl/xdl/config/setup.py)
- 组件实例化： [builder.py](/root/workspace/xdl/xdl/config/builder.py)

## 原先 builder / registry 的主要问题

原先配置系统最关键的几个问题如下：

1. `from_library: torch` 会错误地去 `torch.xxx` 根模块找对象
2. `build_model` 会错误改写 `num_classes`
3. transform pipeline 的非法分支会误导使用者
4. metrics 构建时错误被静默吞掉
5. dataset registry 默认没有稳定的内置数据集可用于 smoke test
6. 官方示例 YAML 混用旧格式和新意图，结构不统一

## 本轮已完成的修复

### 1. 引入 schema-first 配置层

已新增：

- [errors.py](/root/workspace/xdl/xdl/config/errors.py)
- [schema.py](/root/workspace/xdl/xdl/config/schema.py)
- [resolver.py](/root/workspace/xdl/xdl/config/resolver.py)

效果：

- 顶层结构开始收敛到 dataclass
- 未知顶层字段会失败，而不是继续静默漂过去
- `${...}` 插值成为正式能力，而不是 YAML 装饰语法

### 2. builder 从“猜测式”改成“显式式”

当前 builder 同时支持：

- 新格式：`type + source + params`
- 旧格式：`name + from_library + params`

并且修复了库映射：

- `torch` model / loss -> `torch.nn`
- `torch` optimizer -> `torch.optim`
- `torch` scheduler -> `torch.optim.lr_scheduler`
- `torchvision` dataset / transform -> 对应模块路径
- `registry` / `local` -> XDL registry

效果：

- 官方 YAML 的组件来源语义更清晰
- `torch.nn.CrossEntropyLoss`、`torch.optim.Adam`、`torch.optim.lr_scheduler.StepLR` 都能按预期构建

### 3. setup 支持新旧 schema 双轨过渡

当前 [setup.py](/root/workspace/xdl/xdl/config/setup.py) 会：

- 先判断是新 schema 还是旧 schema
- 如果是旧 schema，先归一化到新 schema 结构
- 再 merge 到 dataclass schema
- 再做引用解析
- 最后统一调用 builder 构建组件

效果：

- 官方 YAML 可以直接迁移到 schema v1
- 历史配置暂时不用一步全部推倒重写

### 4. 补齐最小本地可运行组件

当前已新增：

- [basic.py](/root/workspace/xdl/xdl/dataset/basic.py) 中的 `SyntheticClassificationDataset`
- [simple_mlp.py](/root/workspace/xdl/xdl/model/simple_mlp.py) 中的 `SimpleMLP`

效果：

- 官方最小示例可以脱离外部数据集和额外三方依赖稳定构建
- `config/unified_logger_example.yaml` 可以作为 smoke test 示例长期保留

## 当前官方示例状态

目前两个关键示例的定位如下：

- [unified_logger_example.yaml](/root/workspace/xdl/config/unified_logger_example.yaml)
  - 已迁移到 schema v1
  - 可在 CPU 下稳定执行 `setup_from_yaml(...)`
  - 适合作为最小可运行示例

- [vgg_cifar100.yaml](/root/workspace/xdl/config/vgg_cifar100.yaml)
  - 已迁移到 schema v1
  - schema 校验和 model 构建均已通过
  - 完整 `setup_from_yaml(...)` 仍依赖 CIFAR100 数据可用性和下载环境

## 当前已知边界

这轮改造没有回避边界，当前边界建议明确记录：

1. `logging`、`checkpoint`、`accelerate` 虽然已进入 schema，但还没有完全接到 `Trainer` / callback 构建流程。
2. `vgg_cifar100.yaml` 的完整运行依赖真实数据集，不适合作为纯离线单测。
3. 一些本地 dataset 依赖可选三方库，例如 `albumentations`，缺失时不会注册。
4. 当前主要完成的是“配置系统主链路”，不是“完整实验平台闭环”。

## 建议的后续重点

如果继续往下做，优先级建议如下：

1. 为 `logging` / `checkpoint` / `callbacks` 增加 builder，并从 `setup_from_yaml` 或更高层实验入口统一接出。
2. 让 `Trainer` 直接消费 schema v1 的完整实验配置，而不是只消费组件对象。
3. 为官方 YAML 再补 1 到 2 个真实训练级 smoke tests，覆盖 callback / save / logger 主链路。
4. 等基础链路彻底稳定后，再评估是否需要引入更完整的 Hydra 工作流，而不是现在就整体迁移。
