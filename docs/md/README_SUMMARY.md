# XDL MD 摘要入口

本文是 `docs/md/README.md` 的摘要版入口,用于先给 agents 和开发者一个低负担地图,再按需进入详细事实源. 行为,字段,API 和兼容承诺仍以 [README.md](README.md) 及源码为准.

## 这篇摘要负责什么

- 先说明 `docs/md/README.md` 各章节分别解决什么问题.
- 给出推荐阅读顺序和常见改动的跳转入口.
- 提醒哪些主题必须直接看详细版.

## 这篇摘要不负责什么

- 不重复展开安装命令,生命周期细节,完整 schema,API 兼容条款或模块职责表.
- 不单独定义新契约. 需要字段,顺序,边界和兼容细则时,直接打开详细版对应章节.

## 推荐阅读顺序

1. [安装与验证摘要](#安装与验证摘要)
2. [框架与训练摘要](#框架与训练摘要)
3. [配置系统摘要](#配置系统摘要)
4. [Dataset 摘要](#dataset-摘要)
5. [API 与模块边界摘要](#api-与模块边界摘要)
6. [文档与 HTML 规范摘要](#文档与-html-规范摘要)
7. [优化方向摘要](#优化方向摘要)

## 常见改动先看哪里

| 改动范围 | 先看摘要 | 再进详细版 |
| --- | --- | --- |
| 安装,依赖,wheel,运行验证 | [安装与验证摘要](#安装与验证摘要) | [README.md#xdl-安装与验证](README.md#xdl-安装与验证) |
| `Trainer`,`CoreModel`,回调,手动优化 | [框架与训练摘要](#框架与训练摘要) | [README.md#xdl-项目结构与使用说明](README.md#xdl-项目结构与使用说明), [../../xdl/trainer/README.md](../../xdl/trainer/README.md) |
| YAML 配置,schema,`target + params` | [配置系统摘要](#配置系统摘要) | [README.md#xdl-config-系统说明](README.md#xdl-config-系统说明) |
| 数据集模板,manifest,collate | [Dataset 摘要](#dataset-摘要) | [README.md#xdl-dataset-模板规划](README.md#xdl-dataset-模板规划) |
| 公开 API,导出符号,兼容策略 | [API 与模块边界摘要](#api-与模块边界摘要) | [README.md#xdl-api-稳定边界](README.md#xdl-api-稳定边界) |
| 子模块职责或目录迁移 | [API 与模块边界摘要](#api-与模块边界摘要) | [README.md#xdl-模块功能边界速查](README.md#xdl-模块功能边界速查) |
| 文档结构,HTML 视觉系统 | [文档与 HTML 规范摘要](#文档与-html-规范摘要) | [README.md#xdl-html-阅读页样式规范](README.md#xdl-html-阅读页样式规范), [../AGENTS.md](../AGENTS.md) |
| 框架后续待办 | [优化方向摘要](#优化方向摘要) | [README.md#xdl-当前优化方向](README.md#xdl-当前优化方向) |
| XQT 压缩与部署 | [XQT 摘要入口](XQT_SUMMARY.md) | [XQT.md](XQT.md) |

## 安装与验证摘要

你只需要先确认三件事:

- XDL 运行在 Python `>=3.12`, PyTorch `>=1.12`.
- 常规开发安装优先 `pip install -e .`, 需要更多能力再上 `.[all]` 或 `.[all,dev]`.
- 改安装链路后至少验证导入,配置构建主链路和相关测试.

需要具体命令,脚本入口和验证步骤时,直接看 [README.md#xdl-安装与验证](README.md#xdl-安装与验证).

## 框架与训练摘要

XDL 的核心是三层:

- `xdl.config`: 把 YAML 或结构化配置组装成 `TrainSetup`.
- `xdl.trainer`: 承载 `Trainer` / `CoreModel` 生命周期.
- 各注册子模块: `model`,`dataset`,`loss`,`metric`,`optimizer`,`scheduler`,`callbacks`.

改训练链路时先记住:

- 组件通过注册系统集中注册.
- `CoreModel.training_step()` 默认是手动优化模式.
- 生命周期和 batch 迁移规则以 `Trainer` 文档和详细版为准.

需要调用顺序,扩展方式和入口示例时,看 [README.md#xdl-项目结构与使用说明](README.md#xdl-项目结构与使用说明).

## 配置系统摘要

配置主链路遵循:

- 优先 structured config / dataclass schema.
- YAML 组织方式以 `target + params` 为主.
- 加载,合并,插值和容器转换优先交给 `OmegaConf`.

不要在训练入口散落手写 `yaml.safe_load` 合并逻辑. 具体 schema 结构,字段职责和 `setup_from_yaml()` 边界见 [README.md#xdl-config-系统说明](README.md#xdl-config-系统说明).

## Dataset 摘要

数据集相关长期规则只有几条:

- 数据本体统一放 `data/`,不新建平行顶层数据目录.
- dataset 能力通过注册系统接入.
- 模板选择,磁盘组织和扩展流程在详细版统一维护.

改 manifest,loader,collate 或新 dataset 模板时,直接看 [README.md#xdl-dataset-模板规划](README.md#xdl-dataset-模板规划).

## API 与模块边界摘要

写新代码时优先依赖稳定入口:

```python
from xdl.config import setup_from_yaml, TrainSetup
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
from xdl.callbacks import Callback
```

边界判断原则:

- 稳定用户入口看 Stable.
- 正在演进但允许外部试用的入口看 Provisional.
- 内部细节默认按 Internal 管理,不承诺兼容.

具体符号列表和目录职责不要凭印象判断,直接看 [README.md#xdl-api-稳定边界](README.md#xdl-api-稳定边界) 和 [README.md#xdl-模块功能边界速查](README.md#xdl-模块功能边界速查).

## 文档与 HTML 规范摘要

文档分层规则:

- `docs/md/`: 写代码前看的工作文档和事实源.
- `docs/html/`: 给人类读者的阅读版.
- 同一主题先改 MD,再同步 HTML.

HTML 自有页面统一复用 `docs/html/assets/xdl-doc.css` 和 `docs/html/assets/xdl-theme.js`,并遵循固定模板约束. 详细规范见 [README.md#xdl-html-阅读页样式规范](README.md#xdl-html-阅读页样式规范).

## 优化方向摘要

这一节只保留当前仍有效的后续方向,不重复介绍已落地现状. 当你需要判断某个想法是不是当前 roadmap 的一部分,再去看 [README.md#xdl-当前优化方向](README.md#xdl-当前优化方向).

## 何时必须直接看详细版

- 你要改字段名,默认值,兼容行为或 API 导出.
- 你要调整 `Trainer` 生命周期,回调顺序或手动优化语义.
- 你要新增配置 schema,数据集模板或模块边界.
- 你要更新长期文档结构或 HTML 视觉系统.

这类改动不要只看摘要,直接回到 [README.md](README.md).
