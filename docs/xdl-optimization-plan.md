# XDL 优化方向分析

> 基于 2026-04-29 功能边界文档 | 按优先级分三大类

---

## 优先级分类

| 级别 | 含义 | 时间线 | 风险 |
|------|------|--------|------|
| **P0** | 功能缺口 — 已有代码但未接入 | 本周 | 极低 |
| **P1** | 质量提升 — 代码重复/文档/接口 | 本月 | 低 |
| **P2** | 功能增强 — 新增能力 | 下季度 | 中 |

---

## P0 — 立即修复（功能缺口）

### P0-1: 未注册组件补全

**现状**：以下组件已完整实现但未在注册系统中注册，导致无法通过 YAML 配置使用。

| 组件 | 位置 | 行数 | 状态 |
|------|------|------|------|
| `SOAP` 优化器 | `xdl/optimizer/soap.py:10` | 431 行 | 未注册 |
| `MeanAbsoluteError` | `xdl/metric/metrics.py` | ~30 行 | 未注册 |
| `MeanSquaredError` | `xdl/metric/metrics.py` | ~30 行 | 未注册 |
| `RootMeanSquaredError` | `xdl/metric/metrics.py` | ~30 行 | 未注册 |
| `FATT` 分割模型 | `xdl/model/segment/fatt.py:169` | 267 行 | 未注册未导入 |

**修复方案**：在各自 `__init__.py` 的 `_register_*()` 函数中添加注册调用。

**影响范围**：`xdl/optimizer/__init__.py`、`xdl/metric/__init__.py`、`xdl/model/__init__.py`

**工作量**：1 小时

---

### P0-2: `__init__.py` 导出修复

**现状**：以下 5 个回调已完整实现但在 `__init__.py` 中未导出，用户必须通过完整路径导入。

| 回调 | 文件 |
|------|------|
| `DeviceStatsMonitor` | `device_stats_monitor.py` |
| `LambdaCallback` | `lambda_callback.py` |
| `LearningRateMonitor` | `learning_rate_monitor.py` |
| `ModelSummary` | `model_summary.py` |
| `Timer` | `timer.py` |

**修复方案**：在 `xdl/callbacks/__init__.py` 的 `__all__` 和导入中添加这 5 个名称。

**影响范围**：`xdl/callbacks/__init__.py`

**工作量**：10 分钟

---

### P0-3: TwinFlow 缺少文档

**现状**：`xdl/model/generate/twinflow.py` 是整个框架中唯一的生成模型（513 行），完全没有文档字符串。包含复杂的训练步骤（RCGM + 一致性正则化 + 分布匹配）和采样循环（一阶/二阶 ODE 求解器）。

**修复方案**：添加模块级 docstring、类 docstring 和关键方法的 docstring。

**影响范围**：`xdl/model/generate/twinflow.py`

**工作量**：2 小时

---

## P1 — 短期改进（质量提升）

### P1-1: AGENTS.md 更新

**现状**：`xdl/model/AGENTS.md` 缺少以下信息：
- `vit_huge_patch14_224` 工厂函数
- `SimpleMLP` / `simple_mlp`
- 注册表实际计数（写的是示例而非完整 28 条目）
- `FATT` 或 `segment/` 目录的说明
- `TwinFlow` 的使用说明

**修复方案**：更新到与当前代码一致，包含所有 28 个注册条目的完整表格。

**影响范围**：`xdl/model/AGENTS.md`

**工作量**：1 小时

---

### P1-2: 数据集增强代码去重

**现状**：`hairdata.py`、`hairdata3y.py`、`hairdata10hair.py` 三个文件中的 `pixel_transform` 和 `hair_transform` 增强 pipeline 代码几乎完全相同（约 200 行 × 3 = 600 行重复）。

**修复方案**：
1. 在 `xdl/dataset/` 下新建 `transforms.py`
2. 提取 `create_pixel_transform()` 和 `create_hair_transform()` 工厂函数
3. 三个数据集文件导入共享函数

**影响范围**：`xdl/dataset/hairdata.py`、`hairdata3y.py`、`hairdata10hair.py` + 新建 `transforms.py`

**工作量**：3 小时

---

### P1-3: IoU/Dice 多类别接口统一

**现状**：`IoU` 和 `DiceCoefficient` 有两个独立方法：
- `__call__` — 仅支持二分类（阈值二值化）
- `__call_multi_class__` — 支持多类别但需手动调用

**修复方案**：在 `__call__` 中添加 `multi_class=False` 参数，统一接口。

**影响范围**：`xdl/metric/metrics.py`

**工作量**：1 小时

---

### P1-4: Precision/Recall/F1 averaging 模式扩展

**现状**：`Precision`、`Recall`、`F1Score` 硬编码为 `average="macro"`。

**修复方案**：添加 `average` 参数，支持 `"macro"` / `"micro"` / `"weighted"`。

**影响范围**：`xdl/metric/metrics.py`

**工作量**：1 小时

---

### P1-5: 错误处理增强

**现状**：
- Config 系统有完整的异常层次（5 个异常类），但其他模块缺少专用异常
- 数据集文件发现失败时用 `warnings.warn` 而非结构化异常

**修复方案**：
1. 创建 `xdl/exceptions.py` 统一异常层次
2. 数据集模块使用专用异常（`DataDiscoveryError` 等）

**影响范围**：新建 `xdl/exceptions.py` + `xdl/dataset/hairdata*.py`

**工作量**：2 小时

---

## P2 — 中期增强（功能扩展）

### P2-1: 原生 DeepSpeed 集成

**现状**：当前 DeepSpeed 支持完全委托给 HuggingFace Accelerate。无原生 DeepSpeed ZeRO 配置、无 DeepSpeed 检查点格式支持。

**修复方案**：
1. 在 Trainer 中添加 DeepSpeed 初始化路径
2. 支持 DeepSpeed 原生的 `deepspeed.initialize()`
3. 添加 DeepSpeed 检查点格式

**影响范围**：`xdl/trainer/trainer.py`、`xdl/trainer/coreModel.py`、`xdl/config/`

**工作量**：1 周

---

### P2-2: 新增损失函数

**现状**：8 个注册损失函数中 4 个是 PyTorch 原生，只有 FocalLoss 是自定义。

**建议新增**：

| 损失 | 类别 | 优先级 |
|------|------|--------|
| `HuberLoss` | 回归 | 高 |
| `SmoothL1Loss` | 回归 | 高 |
| `InfoNCELoss` | 对比学习 | 中 |
| `DiceLoss` | 分割 | 中 |
| `PerceptualLoss` (VGG-based) | 生成 | 中 |
| `LPIPS` | 生成 | 低 |

**影响范围**：`xdl/loss/` 下新建文件 + `__init__.py`

**工作量**：1-2 周

---

### P2-3: 通用数据集支持

**现状**：4 个数据集：1 个合成测试 + 3 个特定领域图像数据集。缺乏标准学术数据集。

**建议新增**：

| 数据集 | 类别 | 优先级 |
|--------|------|--------|
| `CIFAR10Dataset` | 分类 | 高 |
| `CIFAR100Dataset` | 分类 | 高 |
| `MNISTDataset` | 分类 | 高 |
| `ImageNetDataset` | 分类 | 中 |
| `COCODataset` | 检测 | 低 |

**实现方式**：利用 `torchvision.datasets` 作为后端，做薄包装并接入注册系统。

**影响范围**：`xdl/dataset/` 下新建文件 + `__init__.py`

**工作量**：3-5 天

---

### P2-4: `collate_fn` 支持

**现状**：`build_dataloader()` 仅转发 `**params` 到 `DataLoader`，不支持自定义 `collate_fn`。

**修复方案**：在 `DatasetConfig` schema 中添加 `collate_fn` 字段，支持 `target` 解析。

**影响范围**：`xdl/config/builder.py`、`xdl/config/schema.py`

**工作量**：2 小时

---

### P2-5: 完整测试覆盖

**现状**：经检查 `tests/` 目录，测试覆盖不均衡。

**建议**：

| 模块 | 当前状态 | 目标 |
|------|---------|------|
| callbacks | 基础测试 | 每个回调至少 1 个集成测试 |
| model | 部分 | 每个模型至少 1 个 forward shape 测试 |
| trainer | 基础 | training_step/validation_step 端到端测试 |
| loss | 无 | 每个损失函数数值正确性测试 |
| metric | 无 | 每个指标边界值测试 |
| dataset | 无 | 每个数据集加载测试 |
| optimizer | 无 | Muon 正交化数值测试 |
| config | 部分 | Schema 验证边界测试 |

**工作量**：2-3 周

---

### P2-6: 日志系统增强

**现状**：有 ConsoleCallback + LoggingCallback + TensorBoardCallback + WandbCallback，但缺乏：
- 结构化 JSON 日志格式
- 日志级别动态切换
- 分布式训练时的日志汇聚

**工作量**：1 周

---

## 最迫切优化的优先级排序

综合考虑**影响面、工作量、风险**，推荐按以下顺序执行：

```
P0-1 (SOAP注册)           ← 1小时, 零风险, 立即解锁SOAP
P0-2 (回调导出)            ← 10分钟, 零风险
P0-3 (TwinFlow文档)       ← 2小时, 为最复杂的模型提供可维护性
P1-1 (AGENTS.md更新)      ← 1小时, 为核心模块提供准确导航
P1-2 (数据集增强去重)       ← 3小时, 减少600行重复代码
P1-3 (IoU/Dice统一)        ← 1小时, 改善分割任务用户体验
P0-1续 (指标注册)          ← 30分钟, 解锁3个回归指标
```

**第一批（本周）**：P0-1 + P0-2 — 补全注册缺口，解锁已有功能
**第二批（下周）**：P0-3 + P1-1 + P1-3 — 文档 + 接口修复
**第三批（本月）**：P1-2 + P1-4 + P1-5 — 代码质量提升
**第四批（下月+）**：P2 系列 — 新功能扩展
