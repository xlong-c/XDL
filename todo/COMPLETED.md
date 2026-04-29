# 已完成 — YAML 配置启动体系修复归档

## P0-01: Trainer.fit() 与 TrainSetup 无法衔接

**修复方案**：方案 B — 适配层包装 `nn.Module` → `CoreModel`

**变更**：
- 新建 [xdl/trainer/trainSetupModel.py](../xdl/trainer/trainSetupModel.py)：`TrainSetupModel(CoreModel)` 适配器，将外部 optimizer/loss_fn/scheduler/metrics 注入 CoreModel 的手动优化接口
- [xdl/config/dataclass.py](../xdl/config/dataclass.py)：`TrainSetup.create_model()` 方法，一行创建包装好的 CoreModel
- [xdl/trainer/__init__.py](../xdl/trainer/__init__.py)：导出 `TrainSetupModel`

**用户流**：
```python
setup = setup_from_yaml('config/vgg_cifar100.yaml')
model = setup.create_model()              # nn.Module → CoreModel
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

---

## P0-02: logging / checkpoint / accelerate 配置被静默丢弃

**修复方案**：在 TrainSetup 中增加三个字段，`setup_from_yaml` 提取并传递

**变更**：
- [xdl/config/dataclass.py](../xdl/config/dataclass.py)：新增 `logging_config`、`checkpoint_config`、`accelerate_config` 字段
- [xdl/config/setup.py](../xdl/config/setup.py)：提取三个配置块并传入 TrainSetup 构造函数
- [xdl/trainer/trainer.py](../xdl/trainer/trainer.py)：新增 `Trainer.from_setup()` classmethod，自动将配置传递给 `__init__` 和 `setup_logger()`

---

## P1-03: config_version 声明但不校验

**修复方案**：在 `setup_from_yaml` 入口处校验

**变更**：
- [xdl/config/setup.py](../xdl/config/setup.py)：加载 raw YAML 后立即比较 `config_version` 与 `CONFIG_SCHEMA_VERSION`，不匹配抛 `ConfigValidationError`

---

## P2-04: build_transform 重复调用 _normalize_transform_shorthand

**修复方案**：合并 list/str 分支，结果只 normalize 一次

**变更**：
- [xdl/config/builder.py](../xdl/config/builder.py)：将 `isinstance(config, list)` 和 `isinstance(config, str)` 合并为 `isinstance(config, (list, str))`，normalize 结果直接传入 `_build_component`；Mapping 路径也走一次 normalize 而非浅拷贝

---

## P2-05: _resolve_dataset_value 中存在死代码

**修复方案**：深度修复 — 整个函数简化为直接查询 `built_datasets`

**变更**：
- [xdl/config/setup.py](../xdl/config/setup.py)：`_resolve_dataset_value` 移除 `dataset_value` 和 `built_transforms` 参数，只保留 `dataset_name → built_datasets[dataset_name]` 查询（datasets 在 step 5 已全部预构建，str/dict 分支均为死代码）
- 调用处同步简化

---

## P2-06: 三层收集函数完全重复

**修复方案**：参数化合并为一个 `_collect_configs`

**变更**：
- [xdl/config/setup.py](../xdl/config/setup.py)：删除 `_collect_transform_configs`、`_collect_dataset_configs`、`_collect_dataloader_configs`（共 48 行），统一为 `_collect_configs(root_config, alias_mapping)`（8 行），消除 40 行重复代码

---

## P2-07: setup_from_yaml 内 resolve 流程绕路

**修复方案**：方案 A — 将 resolve 步骤提升到 `load_config_with_schema`，`to_plain_dict` 只做类型转换

**变更**：
- [xdl/config/setup.py](../xdl/config/setup.py)：`load_config_with_schema(raw_config, resolve=True)` + `to_plain_dict(merged_config, resolve=False)`，merge / resolve / convert 三步分离，意图更明确

---

## P2-08: TrainSetup.train_loader 类型标注不准确

**修复方案**：在 `setup_from_yaml` 返回前增加显式校验

**变更**：
- [xdl/config/setup.py](../xdl/config/setup.py)：`train_loader = built_dataloaders.get("train")` 后检查 `if train_loader is None: raise ConfigValidationError(...)`，确保 train_loader 永不为 None，与类型标注一致

---

## P2-09: batch_size fallback 失败时静默回退到 128

**修复方案**：四级 fallback 后仍为 None 时抛 ConfigValidationError

**变更**：
- [xdl/config/setup.py](../xdl/config/setup.py)：移除 `int(selected_batch_size or 128)` 静默回退，新增 `if selected_batch_size is None: raise ConfigValidationError(...)`，要求用户必须显式配置 batch_size

---

## P2-10: Trainer.setup_logger() **kwargs 吞掉未知参数

**修复方案**：将 3 个隐式 kwarg 提升为显式参数，移除 `**kwargs`

**变更**：
- [xdl/trainer/trainer.py](../xdl/trainer/trainer.py)：新增 `console_log_frequency`、`tensorboard_log_frequency`、`save_last` 三个显式参数，移除 `**kwargs`，拼写错误直接 `TypeError` 暴露

---

# 第一轮 P0/P1 — 组件补全 + 质量提升

| # | 任务 | 改动摘要 |
|---|------|----------|
| P0-1 | 未注册组件补全 | +5 注册 (SOAP/MAE/MSE/RMSE/FATT) |
| P0-2 | 回调导出修复 | +5 export (DeviceStats/Lambda/LR/Monitor/Summary/Timer) |
| P0-3 | TwinFlow 文档 | 中文 docstring (模块/类/7 方法) |
| P1-1 | AGENTS.md 更新 | 30 模型表格 + YAML 示例 |
| P1-2 | hairdata 去重 | HairAugMixin, -191 行 |
| P1-3 | IoU/Dice 统一 | __call__ auto-detect 多分类 |
| P1-4 | P/R/F1 average | macro/micro 双模式 |
| P1-5 | 异常层次统一 | XDLError + 5 子类 |

---

# 第二轮 P0/P1 — 功能扩展 + 基础设施

## P0-1: collate_fn 支持

- `xdl/utils/registry.py`：新增 `COLLATE_REGISTRY` + `register_collate`
- `xdl/dataset/collate.py`（新建）：`PadCollate` + `DictCollate`
- `xdl/dataset/__init__.py`：导入 collate 模块
- `xdl/config/builder.py`：`build_collate_fn()` + `build_dataloader(collate_fn=)`
- `xdl/config/schema.py`：schema 新增 `collate_fn` 字段
- `xdl/config/setup.py`：解析并传递 collate_fn

## P0-2: 新增损失函数

- `xdl/loss/huber_loss.py`（新建）：`HuberLoss`
- `xdl/loss/contrastive_loss.py`（新建）：`InfoNCE` / `NTXentLoss`
- `xdl/loss/dice_loss.py`（新建）：`DiceLoss` + `GeneralizedDiceLoss`
- `xdl/loss/__init__.py`：注册 5 个新损失，总数 8 → 13

## P1-1: 视觉数据集模板

- `xdl/dataset/vision_datasets.py`（新建）：`CIFAR10Dataset` / `MNISTDataset`（`_Wrapper` 模式）
- `xdl/dataset/__init__.py`：注册 CIFAR10 / MNIST
- torchvision 可选依赖

## P1-2: DeepSpeed 配置层 (Phase 1)

- `xdl/config/accelerate_config.py`：新增 `DeepSpeedConfig`（from_dict / to_initialize_kwargs）
- `xdl/config/schema.py`：`ConfigSchemaV1` 新增 `deepspeed` 字段
- `xdl/config/dataclass.py`：`TrainSetup` 新增 `deepspeed_config`
- `xdl/config/setup.py`：提取 deepspeed 配置
- `config/deepspeed_zero{1,2,3}.json`（新建）：ZeRO 模板
- API 调研：`model_parameters` 必填；`"auto"` 仅 HF 层支持；与 Accelerate 可共存

## P1-3: 测试覆盖 (Phase 1-2)

- `tests/conftest.py`（新建）：`dummy_batch` / `dummy_image_batch` / `tiny_model`
- `.github/workflows/tests.yml`（新建）：CI（Python 3.10/3.12）
- `tests/utils/test_registry.py`（新建）：9 tests
- `tests/metric/test_metrics.py`（新建）：13 tests
- `tests/loss/test_losses.py`（新建）：14 tests
- `tests/dataset/test_basic.py`（新建）：3 tests
- 测试覆盖：16 (3 files) → 56 (7 files, +40 tests)
