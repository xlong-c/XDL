# XDL 模块功能边界文档

> 基于 2026-04-29 代码库 (commit 35b4bbe) 的完整分析 | 7 个子模块 + 2 个基础设施层

---

## 目录

1. [架构总览](#1-架构总览)
2. [xdl/callbacks — 回调系统](#2-xdlcallbacks--回调系统)
3. [xdl/model — 模型层](#3-xdlmodel--模型层)
4. [xdl/trainer — 训练引擎](#4-xdltrainer--训练引擎)
5. [xdl/loss — 损失函数](#5-xdlloss--损失函数)
6. [xdl/metric — 评估指标](#6-xdlmetric--评估指标)
7. [xdl/dataset — 数据层](#7-xdldataset--数据层)
8. [xdl/optimizer — 优化器](#8-xdloptimizer--优化器)
9. [xdl/scheduler — 学习率调度](#9-xdlscheduler--学习率调度)
10. [xdl/utils — 基础设施](#10-xdlutils--基础设施)
11. [xdl/config — 配置系统](#11-xdlconfig--配置系统)
12. [模块依赖图](#12-模块依赖图)
13. [功能边界矩阵](#13-功能边界矩阵)

---

## 1. 架构总览

```
                    ┌──────────────────────────────────┐
                    │         config/setup.py           │
                    │     setup_from_yaml(config)       │
                    │          ↓ 返回 TrainSetup         │
                    └──────────┬───────────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              │                │                     │
     ┌────────▼──────┐  ┌─────▼──────┐  ┌──────────▼──────┐
     │   Trainer     │  │ TrainSetup │  │  TrainSetupModel │
     │  (编排器)     │  │ (数据类)    │  │  (CoreModel适配) │
     └──────┬────────┘  └────────────┘  └─────────────────┘
            │
    ┌───────┼──────────────────────────────────────────┐
    │       │            CoreModel (基类)                │
    │       │   training_step / validation_step         │
    │       │   configure_optimizers / log / checkpoint │
    └───────┼──────────────────────────────────────────┘
            │
   ┌────────┼────────┬─────────┬─────────┬──────────┐
   ▼        ▼        ▼         ▼         ▼          ▼
Model   Dataset   Optimizer  Scheduler   Loss     Metric
(cnn/   (4个)     (8个)      (10个)     (8个)    (7+3个)
 vit/
 gen)
            │
   ┌────────▼──────────┐
   │  CallbackList      │
   │  15 个回调,27 个钩子 │
   └───────────────────┘
```

### 三种使用模式

| 模式 | 入口 | 适用场景 |
|------|------|---------|
| **纯代码** | `python train_VAE.py` | 研究实验、自定义训练逻辑 |
| **YAML 配置** | `setup_from_yaml('config/xxx.yaml')` | 标准训练任务、快速切换实验 |
| **DeepSpeed/Accelerate** | `deepspeed train_script.py --deepspeed_config ...` | 分布式训练、大模型 |

---

## 2. xdl/callbacks — 回调系统

### 功能边界

**核心职责**：训练生命周期的事件钩子管理。回调观察训练状态但不修改训练逻辑。

**不负责**：
- 不修改模型的训练逻辑（由 CoreModel 子类负责）
- 不管理优化器/调度器步进（由 training_step 负责）
- 不做分布式通信（由 Accelerator 负责）

### 组件清单

```
xdl/callbacks/
├── __init__.py              # 公共 API（仅导出 10 个名称）
├── base.py                  # Callback 基类 (27 个生命周期钩子)
├── callback_list.py         # CallbackList 管理器 (优先级 + 错误隔离)
├── console_callback.py      # ConsoleCallback (priority=200)
├── device_stats_monitor.py   # DeviceStatsMonitor (CPU/GPU/内存监控)
├── early_stopping.py        # EarlyStopping (早停)
├── lambda_callback.py       # LambdaCallback + 3 个便利工厂
├── layer_monitor.py         # LayerMonitor (权重/梯度统计, priority=100)
├── learning_rate_monitor.py # LearningRateMonitor
├── logging_callback.py      # LoggingCallback + SystemStatsCallback
├── model_checkpoint.py      # ModelCheckpoint (支持 pt/st/safetensors)
├── model_summary.py         # ModelSummary (模型结构分析, priority=1)
├── sampling_animation_callback.py # SamplingAnimationCallback (GIF/MP4)
├── tensorboard_callback.py  # TensorBoardCallback (priority=150)
├── timer.py                 # Timer (计时 + ETA)
├── tqdm_callback.py         # TqdmCallback (progress bar, priority=100)
├── wandb_callback.py        # WandbCallback (priority=180)
├── AGENTS.md                # 架构文档
└── README.md                # 开发者指南
```

### 27 个生命周期钩子

| 阶段 | 钩子 |
|------|------|
| **总体** | `setup` / `teardown` / `on_fit_start` / `on_fit_end` |
| **训练** | `on_train_start/end` / `on_train_epoch_start/end` / `on_train_batch_start/end` |
| **验证** | `on_validation_start/end` / `on_validation_epoch_start/end` / `on_validation_batch_start/end` |
| **测试** | `on_test_start/end` / `on_test_epoch_start/end` / `on_test_batch_start/end` |
| **预测** | `on_predict_start/end` / `on_predict_epoch_start/end` / `on_predict_batch_start/end` |
| **检查点** | `on_save_checkpoint` / `on_load_checkpoint` |
| **异常** | `on_exception` |

### 优先级系统

| 优先级 | 回调 | 作用 |
|--------|------|------|
| 1 | ModelSummary | 最先执行，打印模型结构 |
| 100 | TqdmCallback, LayerMonitor | 进度条 + 层监控 |
| 150 | TensorBoardCallback | TensorBoard 日志 |
| 180 | WandbCallback | W&B 日志 |
| 200 | ConsoleCallback | 控制台日志 |
| 500 | SamplingAnimationCallback | 采样动画 |
| 999 (默认) | 其余所有 | 检查点/早停/计时/设备监控 |

### 已知问题
- 5 个回调未在 `__init__.py` 导出：DeviceStatsMonitor, LambdaCallback, LearningRateMonitor, ModelSummary, Timer
- `AGENTS.md` 和 `README.md` 存在但可能过时

---

## 3. xdl/model — 模型层

### 功能边界

**核心职责**：提供可注册的模型架构。模型分为可直接实例化的类和预配置的工厂函数。

**不负责**：
- 不管理训练循环（由 CoreModel/Trainer 负责）
- 不处理权重加载（由 checkpoint 工具 + `load_weight()` 负责）
- 不做分布式包装（由 Accelerator 负责）

### 28 个注册条目

#### CNN 系列

| 注册名 | 类型 | 参数量 | 文件 |
|--------|------|--------|------|
| `ResNet` | 类 | 可变 | `resnet.py:139` |
| `resnet18` | 工厂 | ~11M | `resnet.py:285` |
| `resnet34` | 工厂 | ~22M | `resnet.py:289` |
| `resnet50` | 工厂 | ~25M | `resnet.py:295` |
| `resnet101` | 工厂 | ~45M | `resnet.py:300` |
| `resnet152` | 工厂 | ~60M | `resnet.py:305` |
| `VGG` | 类 | 可变 | `vgg.py:11` |
| `vgg11/11_bn` | 工厂 | ~133M | `vgg.py:130-136` |
| `vgg13/13_bn` | 工厂 | ~133M | `vgg.py:141-147` |
| `vgg16/16_bn` | 工厂 | ~138M | `vgg.py:152-158` |
| `vgg19/19_bn` | 工厂 | ~144M | `vgg.py:163-169` |

#### Vision Transformer 系列

| 注册名 | 类型 | patch/dim | 参数量 | 文件 |
|--------|------|-----------|--------|------|
| `VisionTransformer` | 类 | 可变 | 可变 | `vit.py:133` |
| `vit_tiny_patch16_224` | 工厂 | 16/192 | ~5M | `vit.py:223` |
| `vit_small_patch16_224` | 工厂 | 16/384 | ~22M | `vit.py:231` |
| `vit_base_patch16_224` | 工厂 | 16/768 | ~86M | `vit.py:239` |
| `vit_large_patch16_224` | 工厂 | 16/1024 | ~307M | `vit.py:247` |
| `vit_huge_patch14_224` | 工厂 | 14/1280 | ~632M | `vit.py:255` |

#### 生成模型 / MLP / 组件

| 注册名 | 类型 | 描述 | 文件 |
|--------|------|------|------|
| `TwinFlow` | 类 | 连续时间生成模型 (基于流/扩散) | `generate/twinflow.py:8` |
| `SimpleMLP` / `simple_mlp` | 类/工厂 | 基础 MLP (用于冒烟测试) | `simple_mlp.py:9/33` |
| `BasicBlock` / `Bottleneck` | 类 | ResNet 组件块 | `resnet.py:32/84` |
| `PatchEmbedding` / `MultiHeadAttention` / `TransformerBlock` | 类 | ViT 组件 | `vit.py:12/40/106` |

#### 未注册 (dead code)

| 文件 | 类 | 描述 |
|------|-----|------|
| `segment/fatt.py` | `FATT` | 分割注意力模块，267 行，未注册未导出 |

### 接口规范

所有模型的 `forward` 签名统一为 `forward(self, x: Tensor) -> Tensor`。TwinFlow 是例外（元模型，接收底层 model 作为参数）。

### 已知问题
- `FATT` 存在于 `segment/` 目录但完全未接入系统
- `AGENTS.md` 过时（缺少 `vit_huge_patch14_224`、`SimpleMLP`/`simple_mlp`）
- `TwinFlow` 513 行代码无文档字符串
- 4 个 ViT 组件注册到顶层 MODEL_REGISTRY（可能不应该作为独立条目暴露）

---

## 4. xdl/trainer — 训练引擎

### 功能边界

**核心职责**：编排完整的训练循环，不实现任何具体的模型/损失/优化逻辑。

**不负责**：
- 不实现前向传播/反向传播/优化器步进（由 `CoreModel.training_step()` 负责）
- 不管理数据集加载（由 DataLoader 负责）
- 不管理日志和检查点的具体实现（由 Callback 负责）

### 组件清单

| 类 | 文件 | 职责 |
|----|------|------|
| `Trainer` | `trainer.py:32` | 训练循环编排器 |
| `TrainerState` | `trainer_state.py:11` | 训练状态容器（epoch/step/metrics） |
| `CoreModel` | `coreModel.py:153` | 模型基类（定义 training_step/validation_step 等钩子） |
| `TrainSetupModel` | `trainSetupModel.py:13` | YAML 配置适配器（包装 nn.Module + optimizer + loss） |
| `PipelineInput` | `coreModel.py:22` | 管线输入 dataclass |
| `PipelineOutput` | `coreModel.py:33` | 管线输出 dataclass |
| `ValueDictData` | `coreModel.py:45` | 指标存储容器（支持 batch/epoch 级聚合） |

### 训练循环流程 (伪代码)

```
Trainer.fit(model, train_loader, val_loader):
  1. model.setup("fit")
  2. trainer._setup()           → 设备转移 / Accelerator 准备
  3. trainer._configure_optimizers()
  4. callback_list.setup(trainer, model, "fit")
  5. callback_list.on_train_start()
  6. FOR epoch = 1..max_epochs:
       IF should_stop: BREAK
       state.epoch_start()
       callback_list.on_epoch_start()
       ┌─ _train_epoch():
       │    FOR batch in train_loader:
       │      callback_list.on_train_batch_start()
       │      model.training_step(batch, idx)   ← 用户实现
       │      callback_list.on_train_batch_end()
       └─
       ┌─ IF val_dataloader:  _validate_epoch()
       └─
       IF should_stop: BREAK
  7. callback_list.on_train_end()
  8. callback_list.teardown("fit")
```

### Accelerate 集成

- 当设置 `accelerate_config` 或 `precision` 时激活
- 自动发现模型中所有 `nn.Module` 子模块并通过 `accelerator.prepare()` 包装
- 支持 FSDP：`fsdp=1` (FULL_SHARD) / `fsdp=2` (reshard_after_forward)
- 梯度累积通过 `accelerator.accumulate()` 上下文管理器
- **无原生 DeepSpeed 集成**（完全委托给 Accelerate）

### 已知问题
- 无原生 DeepSpeed 集成路径（依赖 HuggingFace Accelerate 桥接）
- 检查点格式支持 pt/st/accelerator，但不支持 HF safetensors 的完整元数据标准
- `README.md` 内容过时

---

## 5. xdl/loss — 损失函数

### 功能边界

**核心职责**：提供可注册的损失函数。标准损失可用 PyTorch 原生，自定义损失通过注册系统接入。

**不负责**：
- 不做损失加权组合（由 WeightedLoss 包装器负责）
- 不管理优化器步进（由 training_step 负责）

### 8 个注册条目

| 注册名 | 来源 | 类型 |
|--------|------|------|
| `CrossEntropyLoss` | PyTorch 原生 (预注册) | 分类 |
| `MSELoss` | PyTorch 原生 (预注册) | 回归 |
| `L1Loss` | PyTorch 原生 (预注册) | 回归 |
| `BCEWithLogitsLoss` | PyTorch 原生 (预注册) | 二分类 |
| `FocalLoss` | `focal_loss.py:10` | 多分类不平衡 |
| `BinaryFocalLoss` | `focal_loss.py:52` | 二分类不平衡 |
| `focal_loss` | 工厂函数 | 同上 |
| `binary_focal_loss` | 工厂函数 | 同上 |

### 接口规范

所有损失函数遵循 `nn.Module` 接口：`forward(inputs, targets) -> Tensor`

### WeightedLoss 包装器

YAML 配置中当 `loss` 为列表时自动激活。支持多损失加权求和，权重自动归一化。

### 已知问题
- 缺少回归损失（Huber, SmoothL1）
- 缺少对比学习损失（InfoNCE, Triplet）
- 缺少生成模型损失（Perceptual, Adversarial, Diffusion）
- 损失函数仅 2 个自定义实现，大部分依赖 PyTorch 原生

---

## 6. xdl/metric — 评估指标

### 功能边界

**核心职责**：提供可注册的评估指标，每个指标返回 Python float。

**不负责**：
- 不做 epoch 级聚合（由 ValueDictData 负责）
- 不管理指标存储（由 CoreModel.log() 负责）

### 注册条目 (7 个)

| 注册名 | 调用 | 文件 |
|--------|------|------|
| `Accuracy` | `(pred, target) -> float` | `metrics.py:9` |
| `Precision` | `(pred, target) -> float` | `metrics.py:49` |
| `Recall` | `(pred, target) -> float` | `metrics.py:95` |
| `F1Score` | `(pred, target) -> float` | `metrics.py:141` |
| `IoU` | `(pred, target) -> float` | `metrics.py:288` |
| `DiceCoefficient` | `(pred, target) -> float` | `metrics.py:372` |
| `TopKAccuracy` | `(pred, target) -> float` | `metrics.py:258` |

### 已定义但未注册 (3 个)

| 类 | 描述 | 文件 |
|-----|------|------|
| `MeanAbsoluteError` | MAE 回归指标 | `metrics.py` |
| `MeanSquaredError` | MSE 回归指标 | `metrics.py` |
| `RootMeanSquaredError` | RMSE 回归指标 | `metrics.py` |

### 已知问题
- 3 个回归指标已实现但未注册，无法通过 YAML 使用
- IoU 和 DiceCoefficient 的 `__call_multi_class__` 方法与 `__call__` 不统一
- Precision/Recall/F1 仅支持 macro averaging

---

## 7. xdl/dataset — 数据层

### 功能边界

**核心职责**：提供可注册的数据集类，管理数据的加载和预处理。

**不负责**：
- 不做 DataLoader 创建（由 `build_dataloader()` 负责）
- 不做数据增强配置解析（由 `build_transform()` 负责）

### 4 个注册条目

| 注册名 | 数据源 | 输出格式 | 文件 |
|--------|--------|---------|------|
| `SyntheticClassificationDataset` | 随机生成 | `(features, targets)` | `basic.py:11` |
| `GridImageCsvDataset` | CSV + 网格图 | `{source, target, refer}_pixel_values` | `hairdata.py:11` |
| `GridImageDirDataset` | 目录列表 | 同上 | `hairdata3y.py:12` |
| `Hair10HairDataset` | 三文件夹 | 同上 | `hairdata10hair.py:11` |

### 依赖

| 数据集 | torch | albumentations | cv2 |
|--------|-------|---------------|-----|
| SyntheticClassificationDataset | ✅ | - | - |
| GridImageCsvDataset | ✅ | ✅ (可选) | ✅ (可选) |
| GridImageDirDataset | ✅ | ✅ (可选) | ✅ (可选) |
| Hair10HairDataset | ✅ | ✅ (可选) | ✅ (可选) |

### 已知问题
- 3 个图像数据集的增强代码几乎完全相同（约 200 行重复）
- 仅 1 个纯 PyTorch 数据集（用于测试），缺乏通用数据集（CIFAR/MNIST/ImageNet）
- 模块强依赖 albumentations/opencv 作为可选依赖，但无 pure-PyTorch 回退
- 缺少 `collate_fn` 参数支持

---

## 8. xdl/optimizer — 优化器

### 功能边界

**核心职责**：提供可注册的优化器，包括标准 PyTorch 优化器和自定义实现。

**不负责**：
- 不做优化器步进（由 training_step 负责）
- 不做参数分组配置（由 config/builder.py 负责）

### 8 个注册条目

| 注册名 | 来源 | 关键特性 |
|--------|------|---------|
| `Adam` | PyTorch 原生 (预注册) | 标准 Adam |
| `AdamW` | PyTorch 原生 (预注册) | Adam + 解耦权重衰减 |
| `SGD` | PyTorch 原生 (预注册) | 标准 SGD |
| `RMSprop` | PyTorch 原生 (预注册) | RMSprop |
| `Muon` | `muon.py:48` | Newton-Schulz 正交化 + 分布式 all-gather |
| `SingleDeviceMuon` | `muon.py:112` | Muon 的单设备版本 |
| `MuonWithAuxAdam` | `muon.py:151` | 混合优化器 (Muon + AdamW 按参数组切换) |
| `SingleDeviceMuonWithAuxAdam` | `muon.py:261` | 混合优化器单设备版本 |

### 未注册 (已实现)

| 类 | 描述 | 文件 |
|-----|------|------|
| `SOAP` | 基于 Shampoo 预条件器的频谱自适应优化器 (431 行) | `soap.py:10` |

### 已知问题
- `SOAP` 431 行已完整实现但未注册，无法通过 YAML 使用
- Muon 系列优化器缺乏文档

---

## 9. xdl/scheduler — 学习率调度器

### 功能边界

**核心职责**：提供可注册的学习率调度器薄包装。

**不负责**：
- 不做调度器步进决策（由 training_step 负责）

### 10 个注册条目

| 类 | 工厂函数 | PyTorch 基类 |
|----|---------|------------|
| `StepLR` | `step_lr` | `torch.optim.lr_scheduler.StepLR` |
| `MultiStepLR` | `multi_step_lr` | `torch.optim.lr_scheduler.MultiStepLR` |
| `ExponentialLR` | `exponential_lr` | `torch.optim.lr_scheduler.ExponentialLR` |
| `CosineAnnealingLR` | `cosine_annealing_lr` | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| `CosineAnnealingWarmRestarts` | `cosine_annealing_warm_restarts` | `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` |

所有调度器都是纯转发包装器，无自定义逻辑。

### 问题：无已知问题，功能完善。

---

## 10. xdl/utils — 基础设施

### 功能边界

**核心职责**：提供注册系统、检查点工具、权重初始化、模型参数统计。

### 组件清单

| 文件 | 导出 | 职责 |
|------|------|------|
| `registry.py` | `Registry` 类 + 7 个全局实例 + 7 个 `build_*` 函数 | 组件注册与查找系统 |
| `checkpoint.py` | 6 个工具函数 | 检查点保存/加载/格式检测 |
| `weight.py` | 10 个初始化函数 + `update_ema` | 权重初始化 + EMA 更新 |
| `tools.py` | 7 个工具函数 | 路径转换 / 参数统计 / 调试 |

### 7 个全局 Registry

```python
MODEL_REGISTRY       # 28 个条目
DATASET_REGISTRY     # 4 个条目
OPTIMIZER_REGISTRY   # 8 个条目 (4 PyTorch 原生 + 4 Muon)
SCHEDULER_REGISTRY   # 10 个条目
LOSS_REGISTRY        # 8 个条目 (4 PyTorch 原生 + 2 FocalLoss + 2 工厂)
METRIC_REGISTRY      # 7 个条目
TRANSFORM_REGISTRY   # 已定义但未使用
```

### 预注册机制

在 `registry.py` 模块加载时，自动注册 4 个 PyTorch 优化器和 4 个损失函数。这使得 YAML 配置可以直接使用 `target: "registry:Adam"` 等。

### 问题：TRANSFORM_REGISTRY 已创建但未被实际使用。

---

## 11. xdl/config — 配置系统

### 功能边界

**核心职责**：YAML 配置解析、Schema 验证、组件构建、TrainSetup 组装。

**不负责**：
- 不执行训练（由 Trainer 负责）
- 不做运行时状态管理（由 TrainerState 负责）

### 组件清单

| 文件 | 职责 |
|------|------|
| `setup.py` | `setup_from_yaml()` — 顶层编排器 |
| `builder.py` | `build_model/dataset/dataloader/optimizer/scheduler/loss/metrics/transform()` |
| `dataclass.py` | `TrainSetup` 数据类 |
| `schema.py` | OmegaConf Schema 定义 (ConfigSchemaV1, 8 个嵌套 dataclass) |
| `resolver.py` | YAML 加载 + Schema 合并 + `${...}` 插值解析 |
| `errors.py` | 异常层次 (ConfigError → 4 个子类) |
| `loss_weighted.py` | `WeightedLoss` — 多损失组合包装器 |
| `accelerate_config.py` | Accelerate 配置 dataclass |

### Config Target 解析

```
target: "source:ComponentName"

source = "registry"  → REGISTRY.get(ComponentName)
source = "torch.nn"   → importlib → getattr(torch.nn, ComponentName)
source = "torchvision.transforms" → importlib → getattr(torchvision.transforms, ComponentName)
```

### Schema 版本

当前版本 `CONFIG_SCHEMA_VERSION = 1`。

### 问题：无已知问题，配置系统基础设施完善。

---

## 12. 模块依赖图

```
config
  ├──→ utils (registry, checkpoint)
  ├──→ model (build_model)
  ├──→ dataset (build_dataset)
  ├──→ loss (build_loss)
  ├──→ metric (build_metrics)
  ├──→ optimizer (build_optimizer)
  └──→ scheduler (build_scheduler)
        │
trainer
  ├──→ callbacks (CallbackList)
  ├──→ config (TrainSetup, setup_from_yaml)
  └──→ utils (checkpoint)

callbacks
  └──→ trainer (TYPE_CHECKING only)

model, dataset, loss, metric, optimizer, scheduler
  └──→ utils (registry)
```

关键约束：**trainer 和 callbacks 相互引用但通过 TYPE_CHECKING 解开循环依赖**。

---

## 13. 功能边界矩阵

| 模块 | 文件数 | 注册条目 | 外部依赖 | 文档 | 完善度 |
|------|--------|---------|---------|------|--------|
| callbacks | 18 | 15 (隐含) | wandb/tensorboard/psutil | AGENTS.md + README | ⭐⭐⭐⭐⭐ (95%) |
| model | 7 | 28 | torch only | AGENTS.md (过时) | ⭐⭐⭐⭐ (85%) |
| trainer | 5 | N/A | accelerate (可选) | README (过时) | ⭐⭐⭐⭐ (85%) |
| loss | 2 | 8 | torch only | 无 | ⭐⭐⭐ (70%) |
| metric | 2 | 7+3 | torch only | 无 | ⭐⭐⭐ (70%) |
| dataset | 5 | 4 | albumentations/cv2 (可选) | 无 | ⭐⭐ (55%) |
| optimizer | 3 | 8+1 | torch only | 无 | ⭐⭐⭐ (75%) |
| scheduler | 5 | 10 | torch only | 无 | ⭐⭐⭐⭐⭐ (100%) |
| utils | 4 | 7 Registry | torch only | 无 | ⭐⭐⭐⭐⭐ (95%) |
| config | 9 | N/A | OmegaConf/yaml | 无 | ⭐⭐⭐⭐⭐ (95%) |

### 完善度评分说明

- **100%**: scheduler, utils, config — 功能完整，无已知缺口
- **95%**: callbacks — 功能最完善但 5 个回调未导出
- **85%**: model, trainer — 核心功能齐全但有文档缺失和改进空间
- **75%**: optimizer — SOAP 未注册
- **70%**: loss, metric — 基础足够但扩展性弱，有遗漏
- **55%**: dataset — 最薄弱，强外部依赖，缺乏通用数据集

---

## 总结

XDL 框架的**基础设施层（utils + config）设计优秀**，注册系统和 YAML 配置系统是亮点。

**模块间耦合度低**，每个子模块只依赖 utils 层的注册系统，不形成复杂的内部依赖网。

**Trainer 设计遵循委托模式**，将训练逻辑委托给 CoreModel 子类，将观察逻辑委托给 Callback，自身只做编排。

**主要短板**：
1. dataset 模块最弱（依赖重、缺乏通用数据集）
2. 文档覆盖不均衡（callbacks 有完整文档，其余模块几乎没有）
3. 部分已实现组件未接入系统（SOAP/MAE/MSE/RMSE/FATT）
