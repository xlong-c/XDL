# XDL 改进目标与验收

本文承接 [XDL 当前方向](roadmap.md), 把架构评审发现的问题拆成可逐项落地, 可回归验证的任务. 本文是这批 XDL 改进的唯一任务状态源; 方向性描述留在 `roadmap.md`, 不在此重复现状说明.

制定日期: 2026-09-08. 初始状态: 全部任务 `pending`.

## 负责什么

- 记录每个改进任务的证据, 方案和验收标准.
- 维护任务状态和完成记录.
- 给回归测试和文档同步提供统一对齐点.

## 不负责什么

- 不替代源码和既有工程契约.
- 不重复 `roadmap.md` 的方向描述.
- 不记录临时执行日志或一次性实验结果.

## 基线

- 源码基线: `d1e0de0` (master).
- 回归基线: `tests/test_public_api.py tests/config tests/trainer tests/callbacks tests/utils tests/loss tests/metric tests/dataset tests/analysis tests/test_coremodel_logging.py tests/test_framework_fixes.py tests/test_trainer_device_transfer.py` 共 323 项, 全部通过.
- 静态基线: `ruff` 仅启用 `E9, F821`; `pyrightconfig.json` 覆盖 `xdl` 但无 CI 门禁; 无 `.pre-commit-config.yaml`.

## 状态规则

- 状态取值: `pending`, `in_progress`, `blocked`, `done`, `deferred`.
- 任务进入 `done` 需要同时满足: 代码改动完成, 验收命令通过, 对应事实文档同步, 并在第 4 节追加完成记录.
- `deferred` 表示本轮不做, 必须写明原因和恢复条件.

## 1. P0 - 低风险立即可做

### XDL-001 删除 legacy `CoreModel` / `TrainSetupModel` 重复实现

证据:

- `xdl/trainer/coreModel.py` (1,098 行) 与 `xdl/trainer/core_model.py` 是两个不同的类, `xdl.trainer.coreModel.CoreModel is xdl.trainer.core_model.CoreModel` 为 False.
- `xdl/trainer/__init__.py` 导出的是 `core_model`, 旧文件只被同样零引用的 `xdl/trainer/trainSetupModel.py` 引用.
- 旧文件缺少 `configure_unwrapped_modules`, `wait_for_everyone`, `requires_collective_sampling` 和 FSDP state dict 相关方法, 是功能更弱的旧版本.

方案:

- 删除 `xdl/trainer/coreModel.py` 和 `xdl/trainer/trainSetupModel.py`, 不做兼容 shim.

验收:

- `grep -rn "coreModel\|trainSetupModel" --include=*.py xdl/ tests/ train/ examples/` 无命中.
- 公共 API 测试通过.

状态: `done`

### XDL-002 死代码扫描误判澄清

初始证据 (已推翻):

- 初版 AST 扫描认为 `xdl/trainer/example_tasks.py` 和 `xdl/config/loss_weighted.py` 零引用.

实际结论 (2026-09-08 复核):

- `xdl/config/builder.py:476` 在 `build_loss()` 多 loss 分支中函数内 `from .loss_weighted import WeightedLoss`, 是活代码.
- `config/manifest_pair_example.yaml`, `config/manifest_regression_example.yaml`, `config/manifest_segmentation_example.yaml`, `config/manifest_detection_example.yaml` 通过 `target: "xdl.trainer.example_tasks:*"` 引用样例 task.
- 两个文件均已恢复, 不做删除.

方案:

- 不删除. 后续死代码扫描必须覆盖 YAML `target:` 字符串, 函数内 import 和 registry 动态导入字符串.

状态: `withdrawn`

### XDL-003 修正依赖元数据

证据:

- `xdl/trainer/core_model.py` 和 `xdl/trainer/trainer.py` 无条件 `from accelerate import Accelerator`, 但 `pyproject.toml` 把 `accelerate` 放在 optional extra.
- `core_model.py` 使用 `torch.optim.lr_scheduler.LRScheduler` (torch 2.0 引入), 但声明 `torch>=1.12.0`.

方案:

- `accelerate` 提升为核心依赖.
- `torch` 下限提升到 `>=2.0.0`, `torchvision` 同步调整.

验收:

- 纯净环境 `pip install -e .` 后 `from xdl.trainer import CoreModel, Trainer` 成功.

状态: `done`

### XDL-004 修正 checkpoint 保存的入参突变

证据:

- `xdl/utils/checkpoint.py` 的 `save_checkpoint` 使用 `checkpoint.pop("state_dict", {})`, safetensors 分支会改掉调用方传入的 dict.

方案:

- 改为不修改入参的读取方式.

验收:

- 新增回归测试: 保存后原 dict 仍包含 `state_dict`.

状态: `done`

### XDL-005 修正 `Registry.get_signature` 对函数组件失效

证据:

- `sig_obj = obj.__init__ if hasattr(obj, "__init__") else obj` 对所有对象都取 `__init__`; 实测 `get_signature("focal_loss")` 返回 `focal_loss(args, kwargs)`.

方案:

- 函数组件直接取自身签名, 类组件取 `__init__` 签名.

验收:

- 新增回归测试: 函数组件签名包含真实参数名.

状态: `done`

### XDL-006 修正文档漂移

证据:

| 文档 | 声明 | 实际 |
| --- | --- | --- |
| `xdl/trainer/README.md` | `train_setup_model.py` | 实为 `xdl/config/train_setup_model.py` |
| `docs/md/architecture/api-boundary.md` | `from xdl.trainer.train_setup_model import ...` | 路径不存在 |
| `xdl/post_training/AGENTS.md` | `__init__.py::_register_post_training_losses()` | 实为 `_registry.py::register_post_training_losses()` |
| `docs/md/architecture/xdl.md` 分层图 | 缺 `xdl/task`, `xdl/analysis`, `xdl/errors.py` | 已存在 |
| `api-boundary.md` | 8 种注册类型 | 实际 10 种 (含 `CALLBACK`, `TASK`) |

方案:

- 按实际代码修正上述五处.

验收:

- 文档中所有 `from xdl.trainer...` 示例可执行; 注册类型数量与 `xdl/utils/registry.py` 一致.

状态: `done`

### XDL-007 `xdl.task` 顶层懒加载一致性

证据:

- `xdl/__init__.py` 的 `_LAZY_SUBMODULES` 缺 `task`, `import xdl; xdl.task` 抛 `AttributeError`, 其他子包均可用.

方案:

- 把 `task` 加入 `_LAZY_SUBMODULES` 和 `__all__`.

验收:

- 新增回归测试: `import xdl; xdl.task` 可用.

状态: `done`

## 2. P1 - 机制缺陷修复

### XDL-101 回调调度不再吞掉回调体内 `TypeError`

证据:

- `xdl/callbacks/callback_list.py` 捕获回调调用抛出的 `TypeError` 后按签名重试, 回调体内真实 `TypeError` 会导致副作用执行两次, 或原始错误被掩盖.

方案:

- 调用前按签名预过滤 kwargs, 只调用一次; 不再捕获函数体异常.

验收:

- 新增回归测试: 回调体内抛 `TypeError` 时只执行一次且异常被记录.

状态: `done`

### XDL-102 Registry 重名冲突显式化

证据:

- `Registry.register` 遇到重名只 `logger.warning` 后静默保留先注册者.

方案:

- 同名注册不同对象时抛 `RegistryError`; 同名注册同一对象 (重复 import / reload) 保持幂等.

验收:

- 新增回归测试: 重名不同对象报错, 重复注册同一对象不报错.

状态: `done`

### XDL-103 legacy 梯度累积属性优先级显式化

证据:

- `CoreModel.accumulation_steps` 优先读取 `self.gradient_accumulation_steps` 属性, 与 Trainer 注入的 `_gradient_accumulation_steps` 形成双源.
- 复核发现这是被测试保护的 legacy 契约: `tests/test_trainer_device_transfer.py::test_accumulation_helpers_respect_legacy_model_attribute`, 且 `train/posttrain/train_sd35m_apex_xdl.py` 真实使用该属性. 直接删除会造成真实训练入口回归.

方案 (修订):

- 保留 legacy 优先级, 不删除.
- `_set_gradient_accumulation_steps` 在模型属性与 Trainer 注入值不一致时显式告警.

验收:

- 新增回归测试: 冲突时告警且 `accumulation_steps` 以模型属性为准.
- 既有 `test_accumulation_helpers_respect_legacy_model_attribute` 继续通过.

状态: `done`

### XDL-104 修正 `_train_epoch` 早退时的钩子不平衡

证据:

- `remaining_steps <= 0` 时直接 `return`, 跳过了已配对 `epoch_start` 的 `on_epoch_end` / `epoch_end` 回调.

方案:

- 早退路径补齐 epoch 结束钩子.

验收:

- 新增回归测试: 虚拟 epoch 尾部早退时 `on_epoch_end` 仍被调用.

状态: `done`

### XDL-105 `Trainer.test()` 在 Accelerate 路径下准备 test loader

证据:

- `_setup_accelerate` 只 prepare train/val loader, `test()` 直接遍历原始 `self._test_dataloader`, 分布式下每 rank 跑全量.

方案:

- `test()` 在 Accelerate 路径下显式 prepare test loader.

验收:

- 新增回归测试: Accelerate 路径下 test loader 进入 prepare.

状态: `done`

### XDL-106 配置层收敛为单一事实源

证据:

- `TrainSetup` 的扁平字段与 `schema.TrainerConfig` / `RuntimeConfig` 重复; `setup_from_yaml` 在 schema merge 后仍用 `.get(default)` 二次填默认值.
- `resolve_config` 用 `to_container + create` 重建配置, 会丢失 structured config 的 `object_type` 元数据, 导致无法直接取出 dataclass 实例.

方案:

- `TrainSetup` 只保留组件和结构化配置: `trainer` / `runtime` / `logging` / `checkpoint` / `accelerate` / `deepspeed`; 删除 `device` / `num_epochs` / `batch_size` / `precision` / 梯度裁剪 / `fsdp` / `nan_*` / `fail_on_callback_error` / `trainer_config` 等扁平字段.
- `setup_from_yaml` 用 `_to_structured()` 从 schema merge 后的节点直接取 dataclass 实例, 不再二次填默认值; `trainer.batch_size` 写入解析后的 batch size.
- `resolve_config` 改用 `deepcopy + OmegaConf.resolve`, 保留结构化元数据.
- `Trainer.from_setup` 改读结构化字段; `accelerate` 通过 `dataclasses.asdict` 转回 Accelerator 关键字参数.
- 同步更新 `tests/config/test_setup.py` 与 `xdl/{trainer,config}` 文档.

验收:

- `tests/config/` 全部通过; 全量回归 338 项通过; `pyright xdl` 0 errors.

状态: `done`

### XDL-107 移除 resolver 的隐式 `loss` / `metrics` dict 转 list

证据:

- `xdl/config/resolver.py::_normalize_before_merge` 把单 dict 隐式转 list.
- 全仓 YAML 盘点: `config/` 与 `examples/` 中 `loss` / `metrics` 均为 list, 无存量 dict 写法; 文档也未宣传该写法.

方案:

- 删除 `_normalize_before_merge`, 让 schema merge 直接报错; 单 dict 现在抛 `ConfigValidationError`.
- 更新 `tests/config/test_schema.py` 为拒绝用例 (loss / metrics 各一).

验收:

- `test_single_loss_object_is_rejected` / `test_single_metrics_object_is_rejected` 通过.

状态: `done`

## 3. P2 - 中期增强

### XDL-201 `xdl/dataset/hair/` 去任务特化与硬编码

证据:

- 默认路径硬编码 `/root/autodl-tmp`; 注册用 `try/except Exception` 静默降级, `_import_errors` 只写不读.
- 三个数据集文件各自带 `if __name__ == "__main__":` 演示块, 硬编码个人 Windows 路径 (`F:\...`) 并重复定义 `denormalize`.
- 既有 `roadmap.md` 已把 hair 增强逻辑去重列为 P1 方向; 共享 transform 工厂 (`hair_transforms.py`) 已存在.

方案:

- 去掉硬编码 `base_dir` 默认值, 改为必填参数.
- 删除三个数据集文件的 `__main__` 演示块 (演示代码不属于索引/标注逻辑, 同时消除重复 `denormalize` 和个人路径).
- 导入失败改为 `logger.warning` 显式告警, `_import_errors` 保留为可编程查询入口.
- 补 `xdl/dataset/hair/__init__.py` 保持包结构一致, 且不在 `__init__` 主动导入可选依赖.

状态: `done`

### XDL-202 模型接入单一路径

证据:

- `wfen_arch.py`, `dat_arch.py`, `realplksr_arch_ult.py` 只被根目录 `infer/*.py` 直接 import, 不走 registry; `esc_arch.py` 被导入但未注册.
- `xdl/model/__init__.py` 把 `BasicBlock`, `Bottleneck`, `PatchEmbedding` 等内部构件注册为 MODEL.

方案:

- 内部构件 `BasicBlock`, `Bottleneck`, `PatchEmbedding`, `MultiHeadAttention`, `TransformerBlock` 不再注册为 MODEL, 但保留 `xdl.model` 导入面.
- 低层 SR 模型统一走 registry: `RealPLKSR` 直接注册; `RealPLKSR_Ult`, `WFEN`, `DAT_2`, `ESC` 走 `_register_optional_models()` 可选注册, 依赖缺失时告警跳过.
- 移除 `lowlevel/__init__.py` 对 `esc_arch` 的急切导入 (它把 `import xdl.model` 隐式抬到 torch>=2.5).

状态: `done`

### XDL-203 CI 门禁与类型注解

证据:

- `.github/workflows/` 只有 `xdl-jax` 两个 workflow; `ruff` 仅 `E9, F821`; `pyrightconfig.json` 无 CI 调用.

方案:

- 新增 `xdl` CPU CI: `ruff check`, `pytest`, `pyright xdl`.
- 门禁范围内 pyright 存量 20 errors / 3 warnings 已清零; 后续 XDL-205 把范围扩到全 `xdl` 包.

状态: `done`

### XDL-204 Registry 显式 bootstrap 与错误类型

证据:

- `Registry.get()` 隐式 import 整个组件包, 失败时抛原生 `ImportError` 而非 `RegistryError`.

方案:

- bootstrap 失败包装为 `RegistryError`; 显式 bootstrap 入口留待后续评估.

状态: `done`

### XDL-205 全 `xdl` 包 pyright 清零

证据:

- 初始 134 errors, 集中在 `model/lowlevel` 研究模型和 `dataset`; 主要根因是 nn.Module 动态属性未声明, torch/timm stub 差异, 以及研究代码里的 Optional 与类型标注缺失.

方案:

- 按文件修正: 声明 `nn.Module` 动态属性, 用 `getattr`/`cast` 收窄, timm 改从 canonical 子模块导入, 删除研究模型文件里的 `__main__` 演示块.
- CI 门禁从 4 个模块扩到全 `xdl` 包.

验收:

- `python -m pyright xdl` 输出 `0 errors, 0 warnings`.
- 全量回归与导入烟测通过.

状态: `done`

## 4. 完成记录

| 任务 | 状态 | 完成日期 | 证据 |
| --- | --- | --- | --- |
| XDL-001 | done | 2026-09-08 | 删除 `coreModel.py` / `trainSetupModel.py`; 全仓 grep 无命中 |
| XDL-002 | withdrawn | 2026-09-08 | 复核确认两文件均被活代码引用, 见上文 |
| XDL-003 | done | 2026-09-08 | `accelerate` 提为核心依赖; `torch>=2.0.0` |
| XDL-004 | done | 2026-09-08 | `tests/utils/test_checkpoint.py::test_save_checkpoint_does_not_mutate_input` |
| XDL-005 | done | 2026-09-08 | `tests/utils/test_registry.py::test_get_signature_supports_function_components` |
| XDL-006 | done | 2026-09-08 | 修正 6 处文档漂移 (trainer README/AGENTS, api-boundary, xdl.md, post_training AGENTS) |
| XDL-007 | done | 2026-09-08 | `tests/test_public_api.py::test_task_submodule_is_lazily_exported` |
| XDL-101 | done | 2026-09-08 | `tests/callbacks/test_callback_list_errors.py::test_callback_body_type_error_is_not_retried` |
| XDL-102 | done | 2026-09-08 | `tests/utils/test_registry.py` 重名冲突用例 |
| XDL-103 | done | 2026-09-08 | 保留 legacy 优先级 + 冲突告警; `test_legacy_accumulation_attribute_conflict_warns` |
| XDL-104 | done | 2026-09-08 | `test_train_epoch_early_exit_keeps_epoch_hooks_balanced` |
| XDL-105 | done | 2026-09-08 | `test_test_loader_is_prepared_with_accelerate` |
| XDL-201 | done | 2026-09-08 | `tests/dataset/test_hair_contracts.py`; 删除 `__main__` 演示块与硬编码路径 |
| XDL-202 | done | 2026-09-08 | `tests/model/test_model_registry.py`; 构件退出 MODEL registry, 低层 SR 走 registry |
| XDL-203 | done | 2026-09-08 | `.github/workflows/xdl-ci.yml` (ruff + pytest + pyright 门禁) |
| XDL-204 | done | 2026-09-08 | `test_bootstrap_failure_raises_registry_error` |
| XDL-205 | done | 2026-09-08 | `pyright xdl` 0 errors; 17 个文件清零, CI 门禁扩到全包 |
| XDL-106 | done | 2026-09-08 | `TrainSetup` 收敛为结构化配置; `setup_from_yaml` 单一默认值源; `tests/config/` 通过 |
| XDL-107 | done | 2026-09-08 | 删除 `_normalize_before_merge`; `test_single_{loss,metrics}_object_is_rejected` |

全量回归: 338 项通过. `ruff check xdl tests` 与 `pyright xdl` (0 errors) 均通过; CI workflow 见 `.github/workflows/xdl-ci.yml`.

## 关联页面

- [roadmap.md](roadmap.md)
- [xdl.md](xdl.md)
- [module-boundaries.md](module-boundaries.md)
- [api-boundary.md](api-boundary.md)
