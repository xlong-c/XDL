# XDL 包 — 模块化深度学习框架核心

PyTorch 深度学习包，组件注册系统 + 回调生命周期 + YAML 配置化训练。Python 3.8+，PyTorch 1.12+，CUDA 可选。

## 行为准则

- **语言**：所有推理和回答均使用中文
- 做非 trivial 改动前，先用 `grep` / `find` / Agent 搜索代码库，理解现有模式再动手
- 多文件重构或架构级变更时，先进入规划模式（EnterPlanMode）制定方案
- 不确定文件位置或调用关系时，优先搜索而非猜测
- 涉及第三方库用法、API 变更、最佳实践等外部知识时，先用 WebSearch 查最新文档
- **脚本参数**：Python 脚本不使用命令行参数解析库（如 `argparse`/`args`），优先使用 YAML 配置或代码内显式配置

## 架构

```
xdl/
├── __init__.py                    # 包入口
├── errors.py                      # RegistryError 等异常定义
├── config/                        # YAML → TrainSetup 配置系统
│   ├── dataclass.py               # TrainSetup dataclass（一行获取全组件）
│   ├── setup.py                   # setup_from_yaml() 入口
│   ├── builder.py                 # 组件构建器：build_model/build_optimizer/...
│   ├── resolver.py                # 配置解析/变量引用
│   ├── schema.py                  # 配置 schema 版本
│   ├── loss_weighted.py           # WeightedLoss 组合
│   ├── accelerate_config.py       # Accelerate 分布式配置
│   └── errors.py                  # ConfigValidationError 等
├── trainer/                       # 训练器核心
│   ├── trainer.py                 # Trainer 主类
│   ├── coreModel.py               # CoreModel 基类（继承 nn.Module）
│   ├── trainer_state.py           # TrainerState 状态管理
│   └── trainSetupModel.py         # TrainSetupModel（包装外部组件为 CoreModel）
├── callbacks/                     # 训练生命周期钩子（优先级数值越小越先执行，默认 999）
│   ├── base.py                    # Callback 基类
│   ├── callback_list.py           # CallbackList 管理器 + 优先级排序
│   ├── console_callback.py        # 控制台日志
│   ├── tqdm_callback.py           # 进度条（最复杂，952 行）
│   ├── model_checkpoint.py        # 模型检查点
│   ├── early_stopping.py          # 早停
│   ├── tensorboard_callback.py    # TensorBoard
│   ├── wandb_callback.py          # W&B
│   ├── layer_monitor.py           # 层梯度/激活统计
│   ├── learning_rate_monitor.py   # 学习率监控
│   ├── device_stats_monitor.py    # 设备资源监控
│   ├── timer.py                   # 计时器
│   ├── model_summary.py           # 模型结构摘要
│   ├── lambda_callback.py         # 灵活 lambda 钩子
│   ├── logging_callback.py        # 通用日志回调
│   └── sampling_animation_callback.py  # 生成模型采样动画
├── model/                         # 模型架构（30 注册条目）
│   ├── resnet.py                  # ResNet 系列 (18/34/50/101/152)
│   ├── vgg.py                     # VGG 系列 (11/13/16/19 + BN 变体)
│   ├── vit.py                     # Vision Transformer 系列 (Tiny→Huge)
│   ├── simple_mlp.py              # SimpleMLP (784→256→128→10)
│   ├── generate/
│   │   └── twinflow.py            # TwinFlow 连续时间生成模型
│   ├── segment/
│   │   └── fatt.py                # FATT 分割模型
│   └── lowlevel/                  # 底层模型（非注册，直接 import）
│       ├── rgt_arch.py            # RGT (Transformer SR)
│       ├── atd_arch.py            # ATD (Transformer SR)
│       ├── esc_arch.py            # ESC (Transformer SR)
│       ├── rrdb_arch.py           # RRDBNet (ESRGAN backbone)
│       ├── oftsr_unet.py          # OFTSR UNet + SuperResModel
│       └── aesop_autoencoder.py   # AESOP AutoEncoder + ProbabilisticAutoEncoder
├── dataset/                       # 数据集定义
│   ├── basic.py                   # SyntheticClassificationDataset
│   ├── vision_datasets.py         # CIFAR10, MNIST
│   ├── hairdata.py                # GridImageCsvDataset（可选依赖）
│   ├── hairdata3y.py              # GridImageDirDataset（可选依赖）
│   ├── hairdata10hair.py          # Hair10HairDataset（可选依赖）
│   ├── hair_transforms.py         # 毛发数据集 transform
│   └── collate.py                 # PadCollate / DictCollate
├── loss/                          # 损失函数（8 注册条目）
│   ├── focal_loss.py              # FocalLoss / BinaryFocalLoss
│   ├── huber_loss.py              # HuberLoss
│   ├── contrastive_loss.py        # InfoNCE / NTXentLoss
│   └── dice_loss.py               # DiceLoss / GeneralizedDiceLoss
├── metric/                        # 评估指标（10 注册条目）
│   └── metrics.py                 # Accuracy / Precision / Recall / F1Score / IoU / Dice / TopK / MAE / MSE / RMSE
├── optimizer/                     # 优化器（5 注册条目）
│   ├── muon.py                    # Muon / MuonWithAuxAdam 系列
│   └── soap.py                    # SOAP
├── scheduler/                     # 学习率调度器（10 注册条目）
│   ├── step_lr.py                 # StepLR
│   ├── multi_step_lr.py           # MultiStepLR
│   ├── exponential_lr.py          # ExponentialLR
│   ├── cosine_annealing_lr.py     # CosineAnnealingLR
│   └── cosine_annealing_warm_restarts.py  # CosineAnnealingWarmRestarts
└── utils/                         # 工具层
    ├── registry.py                # Registry + 6 种 Registry 实例 + build_*() / register_*()
    ├── checkpoint.py              # save_checkpoint / detect_and_load_checkpoint / flatten/unflatten
    ├── tiling.py                  # tile_inference（分块推理）
    ├── tools.py                   # enable_tensor_debug_info / path_win2wsl
    └── weight.py                  # 权重工具
```

## 核心机制

### 注册系统

六种注册类型，各有独立的 `Registry` 实例和 `register_*` / `build_*` 函数：

| 类型 | Registry 实例 | register 函数 | build 函数 |
|------|-------------|-------------|-----------|
| MODEL | `MODEL_REGISTRY` | `register_model()` | `build_model()` |
| DATASET | `DATASET_REGISTRY` | `register_dataset()` | `build_dataset()` |
| OPTIMIZER | `OPTIMIZER_REGISTRY` | `register_optimizer()` | `build_optimizer()` |
| SCHEDULER | `SCHEDULER_REGISTRY` | `register_scheduler()` | `build_scheduler()` |
| LOSS | `LOSS_REGISTRY` | `register_loss()` | `build_loss()` |
| METRIC | `METRIC_REGISTRY` | `register_metric()` | `build_metric()` |

外加两个辅助注册表：`COLLATE_REGISTRY`、`TRANSFORM_REGISTRY`。

注册模式（在对应 `__init__.py` 中集中调用，**非装饰器模式**）：

```python
# 每个子模块的 __init__.py 中定义 _register_*() 函数集中注册
def _register_losses():
    register_loss("FocalLoss")(FocalLoss)
    register_loss("BinaryFocalLoss")(BinaryFocalLoss)

_register_losses()  # 导入时自动执行
```

### 回调生命周期

```
setup → on_train_batch_end → on_train_epoch_end → on_validation_epoch_end → teardown
```

优先级数值越小越先执行。TqdmCallback(100) < TensorBoard(150) < Wandb(180) < ModelCheckpoint(400) < 默认(999)。

详细文档：[xdl/callbacks/AGENTS.md](xdl/callbacks/AGENTS.md)

### YAML 配置流水线

```
YAML 文件 → setup_from_yaml() → TrainSetup dataclass → create_model() → Trainer.fit()
```

- `config/` 目录存放 YAML 配置文件
- TrainSetup 包含 model / train_loader / optimizer / loss_fn / metrics / scheduler 等全部组件
- builder 模块负责解析 `target: "source:name"` 格式并实例化组件

## 已注册组件速查

### 模型 (30) — 详见 [xdl/model/AGENTS.md](xdl/model/AGENTS.md)

**ResNet**: `BasicBlock`, `Bottleneck`, `ResNet`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
**VGG**: `VGG`, `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`
**ViT**: `PatchEmbedding`, `MultiHeadAttention`, `TransformerBlock`, `VisionTransformer`, `vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224`, `vit_large_patch16_224`, `vit_huge_patch14_224`
**MLP**: `SimpleMLP`, `simple_mlp`
**生成**: `TwinFlow`
**分割**: `FATT`

**注意**：`lowlevel/` 目录下的模型（RGT, ATD, ESC, RRDBNet, OFTSR 系列, AESOP 系列）**不走注册系统**，直接通过 `from xdl.model.lowlevel import XXX` 使用。

### 损失 (8)

`FocalLoss`, `BinaryFocalLoss`, `focal_loss`(工厂), `binary_focal_loss`(工厂), `HuberLoss`, `InfoNCE`, `NTXentLoss`, `DiceLoss`, `GeneralizedDiceLoss`

### 指标 (10)

`Accuracy`, `Precision`, `Recall`, `F1Score`, `IoU`, `DiceCoefficient`, `TopKAccuracy`, `MeanAbsoluteError`, `MeanSquaredError`, `RootMeanSquaredError`

### 优化器 (5)

`Muon`, `SingleDeviceMuon`, `MuonWithAuxAdam`, `SingleDeviceMuonWithAuxAdam`, `SOAP`

### 学习率调度器 (10)

`StepLR`, `step_lr`(工厂), `MultiStepLR`, `multi_step_lr`(工厂), `ExponentialLR`, `exponential_lr`(工厂), `CosineAnnealingLR`, `cosine_annealing_lr`(工厂), `CosineAnnealingWarmRestarts`, `cosine_annealing_warm_restarts`(工厂)

### 数据集 (6)

`SyntheticClassificationDataset`, `CIFAR10`, `MNIST`, `GridImageCsvDataset`, `GridImageDirDataset`, `Hair10HairDataset`

## 开发规范

### 添加新组件

1. 在对应子模块下创建/修改 `.py` 文件
2. 在子模块的 `__init__.py` 中 `import` 并调用 `register_*("Name")(Class)`
3. 在 `__all__` 中导出
4. 所有函数必须加类型注解

### 命名约定

- 类名：`PascalCase`（`ResNet`, `FocalLoss`, `CosineAnnealingLR`）
- 工厂函数：`snake_case`（`resnet50`, `focal_loss`, `cosine_annealing_lr`）
- 注册名：与类名或工厂函数名一致

### 模型必须实现

- `__init__` 接受 `num_classes: int` 参数
- `forward` 标准前向传播
- 在 `xdl/model/__init__.py` 中 import、register、`__all__` 导出

### Loss/Metric 必须实现

- Loss：可调用对象，`__call__(pred, target) → Tensor`
- Metric：继承 `torch.nn.Module`，实现 `update(pred, target)` + `compute() → Tensor`

### 回调必须实现

- 继承 `xdl.callbacks.base.Callback`
- 可选依赖用 `try-except` 包裹，优雅降级
- 不要在回调中修改训练状态（观察者模式）

## 使用方式

```python
# 方式 1：纯代码
from xdl.model import resnet50
from xdl.loss import FocalLoss
from xdl.optimizer import Muon
model = resnet50(num_classes=100)

# 方式 2：YAML 配置（推荐）
from xdl.config.setup import setup_from_yaml
setup = setup_from_yaml("config/resnet_cifar100.yaml")
trainer = Trainer(
    model=setup.model,
    train_dataloader=setup.train_loader,
    val_dataloader=setup.val_loader,
    optimizer=setup.optimizer,
    loss_fn=setup.loss_fn,
    metrics=setup.metrics,
    max_epochs=100,
)
trainer.fit()

# 方式 3：TrainSetup 一键创建
model_trainable = setup.create_model()
trainer = Trainer(model_trainable, ...)
trainer.fit()
```

## 命令

```bash
pip install -e ".[all]"              # 安装
python train_VAE.py                  # VAE 训练 (MNIST)
python train_GAN.py                  # GAN 训练 (MNIST)
CUDA_VISIBLE_DEVICES=0,1 python train_VAE.py  # 多 GPU
pytest tests/ -v --cov=xdl           # 测试
```

## 注意事项

- GPU 内存紧张时用梯度累积、AMP、激活检查点
- DeepSpeed/Accelerate 分布式训练需同步保存检查点
- `lowlevel/` 下模型不走注册系统，不支持 YAML `target: "registry:XXX"` 方式加载
- `learn/` 目录为 C++23 CUDA 内核实验，不遵循 Python 规范
- 数据集模块中的 GridImage/Hair 系列依赖可选第三方库，导入失败不阻断框架
- scheduler 实例化时第一个参数是 optimizer（PyTorch 标准接口）

## 子模块文档

- [xdl/callbacks/AGENTS.md](xdl/callbacks/AGENTS.md) — 回调系统详解
- [xdl/model/AGENTS.md](xdl/model/AGENTS.md) — 模型架构详解
