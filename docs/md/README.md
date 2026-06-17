# XDL MD 工作文档总览

## 第一规则

- `docs/md/` 是给 agents 和开发者写代码前看的工作文档.
- `docs/html/` 是给人类用户阅读的可视化文档.
- 源码和本目录中的 MD 定义事实边界;HTML 只负责把同一套事实讲得更好读.
- 行为,字段,API 或兼容承诺变化时,先改本目录对应 MD,再同步 `../html/` 的阅读页.

研究笔记,阶段性分析和一次性草案应放在 `research/`,不在 `docs/md/` 堆叠.

## 给人看的 HTML

| 文档 | 负责内容 |
| --- | --- |
| [../html/index.html](../html/index.html) | 阅读版总入口,按安装,理解框架,写配置,扩展数据集和 API 边界组织阅读路线. |
| [../html/dataset-structure.html](../html/dataset-structure.html) | 通用 dataset 模块结构图和文件职责说明,不包含 hair 特殊数据集. |
| [../html/xqt.html](../html/xqt.html) | XQT 压缩与部署工具链阅读页,提炼当前模块地图,recipe 状态和验证命令. |
| [../html/style-showcase.html](../html/style-showcase.html) | HTML 阅读页两种固定模板和组件展示,用于维护视觉系统. |

## 给 agents 写代码看的 MD

| 顺序 | 文档 | 负责内容 |
| --- | --- | --- |
| 1 | [INSTALL.md](INSTALL.md) | 安装,验证,当前可运行入口. |
| 2 | [XDL.md](XDL.md) | 框架定位,核心分层,训练入口,扩展方式. |
| 3 | [CONFIG.md](CONFIG.md) | `setup_from_yaml()`,schema v1,`target + params`,YAML 组织方式. |
| 4 | [DATASET.md](DATASET.md) | 数据集语义形态,磁盘组织形态,内置模板和新增 dataset 流程. |
| 5 | [API.md](API.md) | 稳定公共 API,实验性 API,内部实现边界和废弃策略. |
| 6 | [HTML_STYLE.md](HTML_STYLE.md) | 自有 HTML 阅读页统一样式,主题 token,色彩和交互规范. |
| 7 | [xdl-functional-boundary.md](xdl-functional-boundary.md) | 各源码子模块职责速查. |
| 8 | [xdl-optimization-plan.md](xdl-optimization-plan.md) | 当前仍有效的后续优化方向. |
| 9 | [XQT.md](XQT.md) | `xqt/` 模型压缩与部署项目的架构草案,模块拆分,接口草案,recipe backlog 和任务排期. |

## 改动前必读

| 改动范围 | 先读 | 同步要求 |
| --- | --- | --- |
| 公开 API,导出符号,兼容策略 | [API.md](API.md) | 行为变化必须更新 Stable / Provisional / Internal 边界. |
| 训练生命周期,`CoreModel`,`Trainer`,Callback | [XDL.md](XDL.md),[../../xdl/trainer/README.md](../../xdl/trainer/README.md) | 同步说明手动优化,batch 迁移,日志和回调顺序. |
| YAML 配置,schema,`target + params` | [CONFIG.md](CONFIG.md) | 同步字段,示例和配置主链路. |
| 数据集模板,collate,manifest | [DATASET.md](DATASET.md) | 同步模板选择表,注册名和测试要求. |
| 安装,依赖,wheel,运行入口 | [INSTALL.md](INSTALL.md),[../../xdl/USAGE.md](../../xdl/USAGE.md) | 同步安装命令和包内用法入口. |
| 子模块职责,目录迁移 | [xdl-functional-boundary.md](xdl-functional-boundary.md) | 同步目录职责和依赖方向. |
| 文档结构,索引,长期文档边界 | 当前文件,[../AGENTS.md](../AGENTS.md) | 保持 `docs/md/` 与 `docs/html/` 分层清楚. |
| HTML 视觉系统,公共 CSS,主题交互 | [HTML_STYLE.md](HTML_STYLE.md) | 同步 `../html/assets/` 和 HTML 页面引用. |

## 文档边界

- `INSTALL.md` 只讲环境,安装和验证.
- `XDL.md` 只讲框架结构,训练入口和扩展方式.
- `CONFIG.md` 只讲配置系统,不展开 dataset 全量模板规划.
- `DATASET.md` 只讲 dataset 模板规划,选择和扩展方式.
- `API.md` 只讲公共 API 兼容边界,不重复使用教程.
- `HTML_STYLE.md` 只讲自有 HTML 阅读页的视觉系统和样式维护规则,不定义框架行为.
- `xdl-functional-boundary.md` 只做模块职责速查,不重复写长篇使用指南.
- `xdl-optimization-plan.md` 只保留仍然有效的待办,不复述现状说明.
- `XQT.md` 只讲 `xqt/` 压缩与部署工具链规划,不承诺已稳定 API.

## HTML 同步规则

- `../html/index.html` 只做面向用户的阅读入口和文档地图.
- `../html/dataset-structure.html` 只做 dataset 模块结构的可视化说明.
- `../html/xqt.html` 只做 XQT 工具链当前状态的阅读地图,不能替代 [XQT.md](XQT.md) 的任务排期和事实边界.
- HTML 可以重排,提炼和图文化 MD 内容,但不要引入和 MD 或源码冲突的新事实.
- 同一主题的行为,字段或 API 发生变化时,先更新对应 MD,再同步覆盖同一主题的 HTML 页面.
- 新增或重构自有 HTML 时,body 必须且只能包含 `xdl-style-atlas` 或 `xdl-style-ledger` 两种模板之一,默认引用 `../html/assets/xdl-doc.css`;需要交互式主题切换时再引用 `../html/assets/xdl-theme.js`.
- 公共 CSS 只能收敛到 `../html/assets/xdl-doc.css`;目录专属 CSS 只能作为薄入口和局部组件扩展,不能复制公共版式,主题变量或通用阅读组件.
- 新增 CSS 文件前先读 [HTML_STYLE.md](HTML_STYLE.md) 的 CSS 收拢边界,并确认现有入口无法承载.


## 正文收拢说明

本文件现在是 `docs/md/` 的正文事实源. `INSTALL.md`, `XDL.md`, `CONFIG.md`, `DATASET.md`, `API.md`, `HTML_STYLE.md`, `xdl-functional-boundary.md` 和 `xdl-optimization-plan.md` 保留为兼容入口, 只负责把既有链接导向本文对应章节.

## XDL 安装与验证

本文档只讲当前仓库里真实可用的安装方式,验证方式和运行入口,不重复解释框架结构.

### 1. 环境要求

- Python `>=3.12`
- PyTorch `>=1.12`
- Linux / macOS / Windows 均可,当前仓库主要在 Linux 环境下维护
- GPU 不是必需;如果要跑 CUDA 训练,先按本机 CUDA 版本安装对应 PyTorch wheel

### 2. 推荐安装方式

#### 基础安装

适合阅读代码,运行配置构建链路和最小实验:

```bash
git clone https://gitee.com/xlong_t/xdl.git
cd xdl
pip install -e .
```

#### 完整安装

适合需要日志,增强和其他可选能力的环境:

```bash
pip install -e ".[all]"
```

#### 开发环境安装

适合要跑测试和开发检查:

```bash
pip install -e ".[all,dev]"
```

#### 使用仓库脚本

```bash
bash scripts/install.sh base
bash scripts/install.sh full
bash scripts/install.sh dev
```

### 3. PyTorch / CUDA 安装建议

如果本机还没有合适的 PyTorch,先按你的 CUDA 或 CPU 环境安装 wheel,再安装 XDL.

示例:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[all]"
```

```bash
# CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

这一步以你的驱动和 CUDA 环境为准,不建议机械照搬固定版本.

### 4. 安装后怎样验证

#### 验证包导入

```bash
python - <<'PY'
import importlib.metadata
import xdl

print(importlib.metadata.version("xdl"))
print("import xdl ok")
PY
```

#### 查看安装包内用法说明

如果是从 wheel 安装,源码仓库里的 `docs/`,`examples/`,`config/` 不一定存在.安装包内固定提供一份单文件入口:

```bash
python -m xdl.usage
xdl-usage
```

Python 内也可以读取:

```python
import xdl

print(xdl.get_usage_text())
```

#### 验证配置构建主链路

```bash
python - <<'PY'
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
print(type(setup.model).__name__)
print(type(setup.optimizer).__name__)
print(type(setup.loss_fn).__name__)
print(len(setup.train_loader.dataset))
PY
```

如果能正常输出模型,优化器,loss 和数据集长度,说明 schema,resolver,builder 和 `TrainSetup` 主链路已经打通.

#### 跑现有测试

```bash
pytest tests/config -q
```

如果你改动了 registry,dataset,loss,metric,也应额外跑对应子目录测试.

#### 跑开发检查

开发环境使用 Ruff 负责格式化,import 排序和基础 lint, 使用 Pyright/Pylance 负责类型检查:

```bash
ruff check xdl tests config examples infer train research/diffusion-models-survey-2025/sana_hair_lora
pyright
```

需要统一格式时, 再按目录运行 `ruff format <paths>`.

### 5. 当前可直接运行的入口

仓库里当前清晰可见的训练入口主要有:

```bash
python train_VAE.py
python train_GAN.py
python train_TwinFlow.py
```

这些入口代表纯代码路径.

如果要走 YAML 配置路径,当前推荐直接在 Python 中调用:

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

### 6. 可选依赖与现实边界

一些能力依赖可选包:

- `tensorboard`
- `wandb`
- `accelerate`
- `opencv-python`
- `albumentations`
- `loguru`

如果你只做最小验证,不需要一次装满全部依赖;如果要用完整日志或增强链路,优先用:

```bash
pip install -e ".[all]"
```

或:

```bash
bash scripts/install.sh full
```

### 7. 下一步阅读

安装和验证完成后,建议继续看:

1. [README.md](../../README.md)
2. [XDL.md](XDL.md)
3. [CONFIG.md](CONFIG.md)
4. [API.md](API.md)

## XDL 项目结构与使用说明

本文档只回答五件事:

1. XDL 是什么.
2. XDL 的核心层次怎样分工.
3. 当前推荐怎样接入训练.
4. 安装后怎样快速找到用法和公共 API.
5. 新功能应该扩展到哪一层.

### 1. 项目定位

XDL 是一个基于 PyTorch 的模块化深度学习框架.它不替代 PyTorch 的张量,模块和优化器接口,而是在其上补一层更稳定的项目组织能力:

- 组件注册
- YAML 配置构建
- 训练生命周期
- callback 扩展

它更像"项目骨架 + 训练组织层",而不是新的张量计算框架.

### 2. 核心分层

XDL 的主干可以概括为五层:

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

对应关系如下:

- `xdl/model/`:模型与工厂函数
- `xdl/dataset/`:数据集,transform,collate
- `xdl/loss/`:损失函数
- `xdl/metric/`:评估指标
- `xdl/optimizer/`:优化器
- `xdl/scheduler/`:学习率调度器
- `xdl/utils/registry.py`:注册系统
- `xdl/config/`:YAML 到 `TrainSetup`
- `xdl/trainer/`:`CoreModel`,`Trainer`,`TrainSetupModel`
- `xdl/callbacks/`:日志,检查点,进度条,早停等横切逻辑

### 3. 三个关键抽象

#### `Registry`

`xdl/utils/registry.py` 只负责名字到对象的映射.

它不负责:

- 解析 YAML
- 猜测参数
- 拼装训练流程

这一层越简单,越适合做框架稳定基础设施.

#### `CoreModel`

`CoreModel` 承载任务逻辑.新代码推荐从子包入口导入:

```python
from xdl.trainer import CoreModel
```

历史路径 `from xdl.trainer.coreModel import CoreModel` 继续兼容.用户通常需要在子类里实现:

- `training_step()`
- `validation_step()`
- `configure_optimizers()`

XDL 当前采用手动优化模式.也就是说,训练步里需要自行处理:

- `optimizer.zero_grad()`
- `self.manual_backward(loss)`
- `self.clip_gradients(...)`
- `optimizer.step()`
- `self.log("loss", loss, prefix="train")`

简单场景可以使用 `self.manual_optimization_step(loss, ...)` 执行
`zero_grad -> backward(loss / accumulation_steps) -> clip -> step` 模板.

`CoreModel` 会维护训练步计数.新代码优先使用公开属性而不是私有字段:

- `micro_step`:全局 micro-batch 步数,进入 `training_step()` 前已递增.
- `accumulation_steps`:当前梯度累积窗口大小,来自 `Trainer(gradient_accumulation_steps=...)` 或模型侧兼容字段.
- `micro_step_in_accumulation`:当前累积窗口内的 1-based 位置.
- `optimizer_step`:按完整累积窗口推导出的优化器更新次数.
- `is_accumulation_start` / `is_accumulation_boundary` / `should_optimizer_step`:手动累积时判断 `zero_grad()` 与 `optimizer.step()` 的 helper.

日志命名保持旧风格兼容:

- `self.log("train_loss", value)` 继续记录 `train_loss`.
- `self.log("loss", value, prefix="train")` 也记录为 `train_loss`.
- `self.log_metrics({"loss": value}, prefix="val")` 记录为 `val_loss`.
- `value` 支持 Python 数值或单元素 `torch.Tensor`,内部记录为 `float`.
- 已经带 `train_` 或 `train/` 前缀的键不会被重复加前缀.

#### `Trainer`

`xdl/trainer/trainer.py` 负责训练循环编排:

- 设备设置
- epoch / step 循环
- callback 调度
- 验证周期
- 推理采样周期

`Trainer.fit()` 在进入训练循环前会先调用 `model.setup("fit")`.重型模块,外部 pipeline,LoRA 之类惰性初始化逻辑应优先放到这里.

训练,验证,测试 batch 会递归迁移常见容器里的 tensor,支持 `Tensor / dict / list / tuple / dataclass`.第三方自定义对象仍应在 `training_step()` / `validation_step()` 中显式处理设备.

外部 pipeline 或非 `nn.Module` 重组件可通过 `CoreModel.configure_device_objects()` 声明给 Trainer 迁移, 并在 `on_after_device_setup()` 中做任务侧收尾.

### 4. 两条推荐使用路径

#### 4.1 纯代码路径

适合快速研究和高度定制任务.

入口可以参考:

- [train_VAE.py](../../train_VAE.py)
- [train_TwinFlow.py](../../train_TwinFlow.py)

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

#### 4.2 YAML 配置路径

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

这里的职责分工是:

- `setup_from_yaml()`:解析配置并构建组件
- `TrainSetup`:承载 `model / optimizer / dataloader / loss / metrics`
- `setup.create_model()`:把外部组件包装成 `TrainSetupModel`
- `Trainer.from_setup()`:按配置补齐常用回调

### 5. 安装后单文件入口和公共 API

只安装 wheel,没有源码仓库时,可以直接查看包内用法摘要:

```bash
python -m xdl.usage
xdl-usage
```

Python 内也可以读取同一份文本:

```python
import xdl

print(xdl.get_usage_text())
```

新代码优先依赖这些稳定入口:

```python
from xdl.config import setup_from_yaml, TrainSetup
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
from xdl.callbacks import Callback
```

完整 Stable / Provisional / Internal API 边界见 [API.md](API.md).

### 6. 当前推荐的接入顺序

当你要接一个新训练任务时,优先按下面顺序阅读:

1. [train_VAE.py](../../train_VAE.py)
2. [train_TwinFlow.py](../../train_TwinFlow.py)
3. [xdl/trainer/trainer.py](../../xdl/trainer/trainer.py)
4. [xdl/trainer/coreModel.py](../../xdl/trainer/coreModel.py)
5. [CONFIG.md](CONFIG.md)
6. [API.md](API.md)

这样能最快看清真实生命周期,而不是只看目录名猜结构.

### 7. 怎样扩展 XDL

#### 新增模型 / 数据集 / loss / metric / optimizer / scheduler

规则统一:

1. 在对应子模块中实现类或工厂函数
2. 在对应 `__init__.py` 中集中注册
3. 通过 `register_*("Name")(Class)` 接入
4. 在 `__all__` 中导出

不要在定义处直接用装饰器式注册;本仓约定是集中注册.

#### 新增 callback

放到 `xdl/callbacks/`,继承 `Callback`,处理自己的生命周期钩子.回调优先作为观察者存在,不要把主要训练逻辑塞进去.

#### 新增配置能力

优先判断改动属于哪一层:

- 顶层结构变化:改 `schema.py`
- 配置解析变化:改 `resolver.py`
- 组件实例化变化:改 `builder.py`
- 训练装配变化:改 `setup.py`

### 8. 当前边界

XDL 现在已经具备稳定的"组件注册 + 配置构建 + 训练编排"主链路,但仍有明确边界:

- 不是统一 CLI 框架
- 不是完整实验平台
- 不是自动优化框架
- 分布式和外部大模型接入仍偏工程化

这也是为什么文档和代码都强调"看真实训练入口",不要假设存在一条包办一切的自动主流程.

## XDL Config 系统说明

本文档只讲 XDL 当前的配置系统,不重复介绍整个框架结构.

### 1. 配置系统解决什么问题

XDL 的 config 系统负责把 YAML 变成可运行组件集合.它解决的是:

- 固定实验配置的上层结构
- 用统一写法构建模型,数据集,优化器,调度器,loss,metrics
- 把配置解析和对象实例化从训练脚本里抽出来

它不负责替代 `Trainer`,也不试图把全部训练逻辑都塞进 YAML.

### 2. 当前主链路

当前推荐入口是:

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
```

主链路如下:

```text
YAML
  -> load_config_with_schema()
  -> to_plain_dict()
  -> build_model / build_dataset / build_dataloader
  -> build_optimizer / build_scheduler / build_loss / build_metrics
  -> TrainSetup
```

返回结果是 `TrainSetup`,其中包含:

- `model`
- `train_loader`
- `val_loader`
- `test_loader`
- `optimizer`
- `scheduler`
- `loss_fn`
- `metrics`
- `logging_config`
- `checkpoint_config`
- `accelerate_config`
- `trainer_config` 以及 `precision` / `gradient_accumulation_steps` 等常用 trainer 字段

如果要直接交给 `Trainer.fit()`,可以继续:

```python
model = setup.create_model()
```

公共入口契约见 [API.md](API.md).配置主链路推荐只依赖 `from xdl.config import setup_from_yaml, TrainSetup`,不要直接依赖 `xdl.config.setup` 内部辅助函数.

### 3. 模块分工

`xdl/config/` 主要由四部分组成:

- [schema.py](../../xdl/config/schema.py):固定顶层结构和默认值
- [resolver.py](../../xdl/config/resolver.py):schema merge,插值解析,普通 dict 转换
- [builder.py](../../xdl/config/builder.py):组件实例化
- [setup.py](../../xdl/config/setup.py):组装整条构建链路

职责边界如下:

- schema 决定"允许什么结构"
- resolver 决定"配置怎样被解析"
- builder 决定"对象怎样被构建"
- setup 决定"怎样把对象拼成 `TrainSetup`"

### 4. 顶层结构

当前 schema 版本为 `v1`,顶层字段包括:

- `config_version`
- `runtime`
- `trainer`
- `model`
- `task`
- `train_transforms` / `val_transforms` / `test_transforms`
- `train_dataset` / `val_dataset` / `test_dataset`
- `dataloader_defaults`
- `train_dataloader` / `val_dataloader` / `test_dataloader`
- `optimization`
- `loss`
- `metrics`
- `callbacks`
- `logging`
- `checkpoint`
- `accelerate`
- `deepspeed`
- `xdl`

其中最常用的几块是:

- `runtime`:设备,实验名,输出目录
- `trainer`:epoch,batch size,precision,梯度累积
- `model`:模型组件
- `task`: 可选的 `CoreModel` 任务组件, 适合大模型或手写训练逻辑
- `optimization`:优化器和调度器
- `loss` / `metrics`:训练目标和评估指标
- `callbacks`: 用 import path 配置化构建 callback

### 5. 统一组件写法

XDL 当前统一使用 `target + params`:

```yaml
model:
  target: "registry:simple_mlp"
  params:
    input_size: 784
    hidden_size: 128
    num_classes: 10
```

`target` 使用 `source:name` 形式.

常见来源:

- `registry:...`
- `torch.nn:...`
- `torch.optim:...`
- `torch.optim.lr_scheduler:...`
- `torchvision.transforms:...`

`builder.py` 会先解析 `target`,再实例化组件.

内置上下文字段由配置加载器注入:

```yaml
runtime:
  output_dir: ${xdl.abspath:${xdl.config_dir},outputs}
```

可用字段:

- `${xdl.config_path}`: 当前 YAML 文件的绝对路径
- `${xdl.config_dir}`: 当前 YAML 文件所在目录
- `${xdl.project_root}`: 调用 `setup_from_yaml()` 时的当前工作目录
- `${xdl.join_path:...}` / `${xdl.abspath:...}`: 路径拼接和绝对路径 resolver

### 6. transform,dataset,dataloader 的关系

配置系统把数据链路拆成三层:

1. transform
2. dataset
3. dataloader

示例:

```yaml
dataloader_defaults:
  batch_size: ${trainer.batch_size}
  num_workers: 0
  pin_memory: false

train_dataset:
  target: "registry:SyntheticClassificationDataset"
  params:
    num_samples: 500
    input_shape: [784]
    num_classes: 10
    seed: 42

train_dataloader:
  dataset: ${train_dataset}
  params:
    shuffle: true
    drop_last: true
```

当前实现里:

- `dataloader_defaults` 负责公共默认值
- `train_dataloader.params` 负责局部覆盖
- `val_dataloader.dataset: ${train_dataset}` 这类写法可以复用已有 dataset 配置
- `trainer.batch_size` 可作为默认 batch size 来源
- `collate_fn` 支持 `None`,可调用对象或 `target + params`

完整 dataset 模板规划, 选择表和新增 dataset 流程见 [DATASET.md](DATASET.md). 本节只保留配置系统需要知道的写法.

#### Dataset 配置边界

配置文档只说明 dataset 怎样进入 YAML 构建链,不展开完整模板规划. 完整 dataset 模板选择,字段约定,磁盘组织和新增流程见 [DATASET.md](DATASET.md).

配置侧需要遵守三条规则:

- dataset 本身仍使用 `target + params`.
- dataloader 通过 `dataset: ${train_dataset}` 复用 dataset 配置,再用 `params` 覆盖 batch 行为.
- `collate_fn` 需要特殊处理时也使用 `target + params`,例如 detection 或 image edit.

Manifest 分类最小示例:

```yaml
train_dataset:
  target: "registry:RecordClassificationDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_cls.jsonl}
    transform: ${train_transforms}
```

Detection 自定义 collate 示例:

```yaml
train_dataloader:
  dataset: ${train_dataset}
  collate_fn:
    target: "registry:DetectionCollate"
  params:
    batch_size: 4
    shuffle: true
```

历史 `registry:Manifest*` 名称继续可用,但新配置优先写 `registry:Record*` 或 `registry:ImageEdit*`.

### 7. loss 和 metrics 的写法

#### 单个 loss

```yaml
loss:
  - target: "torch.nn:CrossEntropyLoss"
    params: {}
```

#### 多个 loss

`build_loss()` 支持列表形式,多项时会构建 `WeightedLoss`:

```yaml
loss:
  - target: "torch.nn:MSELoss"
    params: {}
    weight: 1.0
  - target: "registry:DiceLoss"
    params: {}
    weight: 0.5
```

#### metrics

```yaml
metrics:
  - target: "registry:Accuracy"
    params:
      num_classes: 10
```

`build_metrics()` 返回指标实例列表.

#### callbacks

Callback 暂不新增 registry 类型, 使用 import path 构建:

```yaml
callbacks:
  - target: "xdl.callbacks:SaveTrainableStateCallback"
    params:
      dirpath: ${xdl.abspath:${xdl.config_dir},adapters}
      every_n_epochs: 1
```

`Trainer.from_setup(setup)` 会把这些 callback 加入训练器.

#### dataset 参数中的可调用 transform

除了 `transform` / `target_transform`, 内置 manifest 文本模板还支持把
`text_transform` 和 `target_text_transform` 写成 `target + params` 组件配置:

```yaml
train_dataset:
  target: "registry:RecordTextDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_text.jsonl}
    text_transform:
      target: "torch.nn:Identity"
      params: {}
```

#### CoreModel task

标准监督任务继续使用 `model + optimization + loss`. 如果任务本身继承
`CoreModel` 并在 `configure_optimizers()` 中构建优化器, 可以改用:

```yaml
task:
  target: "my_project.tasks:MyTask"
  params:
    lr: 0.0001
```

此时 `optimizer` 和 `loss` 可以省略, `setup.create_model()` 会直接返回该
`CoreModel` 实例.

### 8. 一个最小可运行示例

仓库里的 [config/unified_logger_example.yaml](../../config/unified_logger_example.yaml) 是当前最合适的主路径样例.

如果想看 manifest 数据模板的完整官方样例, 参考:

- [config/manifest_segmentation_example.yaml](../../config/manifest_segmentation_example.yaml)
- [config/manifest_detection_example.yaml](../../config/manifest_detection_example.yaml)
- [config/manifest_regression_example.yaml](../../config/manifest_regression_example.yaml)
- [config/manifest_pair_example.yaml](../../config/manifest_pair_example.yaml)

最小消费方式:

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")

print(type(setup.model).__name__)
print(type(setup.optimizer).__name__)
print(type(setup.loss_fn).__name__)
print(len(setup.train_loader.dataset))
```

### 9. 推荐扩展方式

当你想扩展配置系统时,先判断变化属于哪类:

- 新顶层字段:改 `schema.py`
- 新插值或 merge 规则:改 `resolver.py`
- 新组件构建方式:改 `builder.py`
- 新装配流程:改 `setup.py`

不要把所有变化都堆到 `setup_from_yaml()`.

训练入口里的轻量 dataclass 配置推荐使用:

```python
from xdl.config import load_structured_dataclass_config

config = load_structured_dataclass_config(MyConfig, "train.yaml")
```

该工具遵循 `structured dataclass 默认值 -> YAML 覆盖 -> overrides 覆盖`
的顺序, 并统一走 OmegaConf resolver 和插值解析.

### 10. 当前边界

当前 config 系统已经稳定支持:

- schema v1 顶层结构
- `target + params`
- transform / dataset / dataloader 构建
- optimizer / scheduler / loss / metrics 构建
- `TrainSetup` 返回

但仍有边界:

- 它不替代训练主循环
- 它不定义统一 CLI
- 它不自动覆盖所有任务特化逻辑
- 它更适合"配置化构建组件",而不是"声明式描述整个实验世界"
- schema dataclass,resolver 和 builder 内部辅助函数仍属于演进中的 API;稳定入口以 [API.md](API.md) 为准

## XDL Dataset 模板规划

本文档负责说明 XDL 数据集模板怎样选择, 怎样扩展, 以及新增数据集时优先复用哪些基类和模板. 配置系统写法只保留必要示例, 完整 `target + params` 规则见 [CONFIG.md](CONFIG.md).

### 1. 规划原则

设计 dataset 时先分清两个维度:

- 样本语义形态: 单条样本返回哪些字段, 例如 `image + label`, `image + mask`, `image + text`, `pair`, `triplet`, `text + target_text`.
- 磁盘组织形态: 数据在磁盘上怎样摆放, 例如 manifest, 目录分类, basename sidecar, 纯图片目录, 标准标注格式.

XDL 的推荐顺序是:

1. 能写 manifest 时, 优先使用 `Record*` 模板. 字段清晰, 可扩展, 容易跨机器复现.
2. 数据已经是目录分类或纯图片目录时, 使用目录模板, 不强制先生成 manifest.
3. 数据是同名文件对齐时, 使用 sidecar 模板, 例如 `images/000.png` 对应 `masks/000.png` 或 `000.txt`.
4. 只有当字段解析, 采样逻辑或外部格式确实特殊时, 才新增薄 dataset 类.
5. 新增模板必须通过 `xdl.dataset.__init__` 集中注册, 并至少覆盖 `__len__`, `__getitem__`, registry 构建和基本 batch 行为测试.

命名约定: 新代码优先使用 `Record*` / `ImageEdit*` dataset 名称. 历史 `Manifest*` 类名和 registry 键仍保留为兼容别名, 例如 `ManifestRegressionDataset` 仍等价于 `RecordRegressionDataset`.

### 2. 样本语义形态

| 语义形态 | 典型任务 | 推荐模板 |
|---|---|---|
| `image` | 推理, 自监督, 特征提取 | `ImageFolderDataset` |
| `image + label` | 单标签分类 | `ImageFolderClassificationDataset`, `RecordClassificationDataset` |
| `image + target` | 回归, 质量估计, 年龄预测 | `RecordRegressionDataset` |
| `image + labels` | 多标签分类 | `RecordMultiLabelClassificationDataset` |
| `image + mask` | 语义分割, depth, 逐像素标注 | `RecordSegmentationDataset`, `ImageMaskSidecarDataset` |
| `image + boxes + labels` | 目标检测 | `RecordDetectionDataset` |
| `image + text` | caption, diffusion 微调, 图文检索 | `RecordImageTextDataset`, `ImageTextSidecarDataset` |
| `text (+ target_text)` | 文本分类前处理, SFT, 指令数据 | `RecordTextDataset` |
| `image_a + image_b (+ label)` | pair matching, siamese, 对比学习 | `RecordPairDataset` |
| `anchor + positive + negative` | metric learning, retrieval | `RecordTripletDataset` |
| `source_image + target_image (+ reference_image + edit_mask + prompt)` | 图像编辑, 条件生成 | `ImageEditDataset` |
| 任意 dict record | 表格, 推荐, 时序窗口, 私有 schema | `RecordDataset` 或继承 `RecordDatasetBase` |

### 3. 磁盘组织形态

#### Manifest 主路径

Manifest 支持 `.jsonl`, `.json`, `.csv`, 相对路径默认基于 manifest 所在目录解析.

```json
{"image": "images/000.png", "label": "cat", "sample_id": "000"}
{"image": "images/001.png", "label": "dog", "sample_id": "001"}
```

适合:

- 长期训练数据
- 多字段样本
- 多任务数据
- 分布式训练前的数据快照
- 需要固定 schema 和复现的数据

新增 manifest 数据集时优先继承 `RecordDatasetBase`, 复用:

- `load_manifest_context()`
- `self.records`
- `self.base_dir`
- `self._record_at(index)`
- `self._resolve_record_path(record, key)`
- `self._sample_id_from_path(...)`
- `self._sample_id_from_fallback(...)`

#### 目录分类

```text
root/
  cat/
    000.png
  dog/
    001.png
```

使用 `ImageFolderClassificationDataset`, 返回 `(image, target)`.

#### 纯图片目录

```text
images/
  000.png
  nested/001.jpg
```

使用 `ImageFolderDataset`, 返回:

```python
{
    "image": image,
    "sample_id": "000",
    "image_path": "...",
}
```

#### Image + text sidecar

```text
data/
  000.png
  000.txt
```

或:

```text
images/
  000.png
texts/
  000.txt
```

使用 `ImageTextSidecarDataset`, 返回 `image`, `text`, `sample_id`, 可选路径字段.

#### Image + mask sidecar

```text
images/
  000.png
masks/
  000.png
```

或同目录不同扩展名:

```text
data/
  000.jpg
  000.png
```

使用 `ImageMaskSidecarDataset`, 返回 `image`, `mask`, `sample_id`, 可选路径字段. 它与 `RecordSegmentationDataset` 返回结构一致, 因此下游 segmentation task 可以少改或不改.

同目录 sidecar 模式建议显式设置 `extensions`, 用来只扫描 image 文件, 避免把 mask 文件再次当作 image 样本.

#### 标准格式适配

COCO, YOLO, VOC, keypoint 等标准格式目前建议先转换为 manifest. 后续如果某个格式在多个项目中反复出现, 再新增专门 dataset 适配器, 但仍应把返回字段对齐到 `image + boxes + labels`, `image + mask`, 或 keypoint 的稳定 dict 结构.

### 4. 加载模式

| 加载模式 | XDL 推荐用法 | 适用场景 |
|---|---|---|
| Map-style dataset | 当前所有内置模板主路径 | 文件数量明确, 可随机访问 |
| Iterable dataset | 暂不作为内置主模板 | 流式日志, 超大文本, remote shard |
| 懒加载 | 图像, mask, text sidecar 模板默认采用 | 大多数图片, 分割, 检测数据 |
| 全量内存 | 自定义 dataset 或 transform 内处理 | 小型表格, toy data |
| 预处理缓存 | 先生成 `.pt`, `.npy`, `.parquet`, `.jsonl` manifest | tokenizer 结果, latent cache, embedding cache |
| 动态 transform | dataset 参数传 `transform` | 图像增强, 同步 mask/box 变换 |
| 自定义 collate | `collate_fn.target: registry:...` | detection, text padding, 复杂 dict batch |
| 分布式采样 | `DataLoader`/训练入口层处理 | 多卡训练, 大规模数据 |

当前内置模板重点覆盖 map-style + 懒加载 + manifest/sidecar/目录组织. Iterable/shard/远程读取可以通过继承 `torch.utils.data.IterableDataset` 单独实现, 但不要把流式逻辑硬塞进 manifest 模板.

### 5. Transform 和 Collate

单输入图像任务可以直接使用普通 image transform.

`image + mask` 使用 `ImageMaskTransform`, 它会对 image 和 mask 共享 resize/crop/flip 决策, 并保证 mask 用 nearest 语义.

`image + boxes` 使用 `ImageBoxesTransform`, 它会同步 resize/flip boxes.

`DetectionCollate` 保留每张图不同数量的 `boxes` 和 `labels`.

`DictCollate` 适合大多数 dict 样本, tensor 字段会尝试 stack, stack 失败时保留 list.

`PadCollate` 适合简单变长序列 padding. 复杂 NLP token padding 更推荐在项目侧写专门 collate 或 tokenizer wrapper.

### 6. YAML 示例

Manifest 分类:

```yaml
train_dataset:
  target: "registry:RecordClassificationDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_cls.jsonl}
    transform: ${train_transforms}
```

Image + mask sidecar:

```yaml
train_dataset:
  target: "registry:ImageMaskSidecarDataset"
  params:
    image_root: ${xdl.abspath:${xdl.config_dir},data/images}
    mask_root: ${xdl.abspath:${xdl.config_dir},data/masks}
    transform:
      target: "registry:ImageMaskTransform"
      params:
        height: 512
        width: 512
        random_flip: true
```

同目录不同扩展名 mask:

```yaml
train_dataset:
  target: "registry:ImageMaskSidecarDataset"
  params:
    root: ${xdl.abspath:${xdl.config_dir},data/image_mask}
    extensions: [".jpg"]
    mask_extension: ".png"
    transform:
      target: "registry:ImageMaskTransform"
      params:
        height: 512
        width: 512
```

Detection 需要自定义 collate:

```yaml
train_dataloader:
  dataset: ${train_dataset}
  collate_fn:
    target: "registry:DetectionCollate"
  params:
    batch_size: 4
    shuffle: true
```

### 7. 新增数据集流程

新增数据集时按下面顺序判断:

1. 字段已经能用 manifest 表达: 直接用现有 manifest 模板, 或继承 `RecordDatasetBase` 只实现 `__getitem__`.
2. 只是路径组织不同: 优先补 sidecar/目录扫描 helper, 不要复制完整 dataset 类.
3. 只是 batch 方式不同: 新增 collate, 不要改 dataset 返回结构.
4. 只是单样本增强不同: 新增 transform, 不要新增 dataset.
5. 确实是新语义形态: 新增 dataset 文件, 在 `xdl/dataset/__init__.py` 注册, 更新 `docs/md/DATASET.md`, `docs/md/API.md`, `xdl/dataset/AGENTS.md`, 并补测试.

薄 subclass 示例:

```python
from typing import Any, Dict

from xdl.dataset import RecordDatasetBase


class MyImageJsonDataset(RecordDatasetBase):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        base_index, record = self._record_at(index)
        image_path = self._resolve_record_path(record, "image")
        json_path = self._resolve_record_path(record, "annotation")
        return {
            "image_path": str(image_path),
            "annotation_path": str(json_path),
            "sample_id": self._sample_id_from_path(record, image_path, base_index),
        }
```

如果这个类要进入 XDL 内置组件, 还需要在 `xdl/dataset/__init__.py` 中集中注册:

```python
register_dataset("MyImageJsonDataset")(MyImageJsonDataset)
```

### 8. 验证清单

新增或改动 dataset 后至少验证:

- `len(dataset)` 与样本数量一致.
- `dataset[0]` 返回字段名和类型稳定.
- 相对路径基于 manifest 或 root 正确解析.
- 缺失必需字段或 sidecar 文件时错误清楚.
- `DataLoader(dataset, batch_size=2)` 或指定 collate 能正常产出 batch.
- registry 路径 `build_dataset({"target": "registry:Name", "params": ...})` 可构建.
- 文档中的 YAML 字段名和构造参数一致.

## XDL API 稳定边界

本文档定义 XDL 对外承诺的 Python API 边界.它不替代使用教程,也不列出每个模型或损失函数的参数;这些细节仍以源码,配置样例和组件文档为准.

### 稳定级别

XDL API 分为三类:

- Stable:推荐用户和训练脚本直接依赖.兼容性变更需要保留旧路径并给迁移说明.
- Provisional:可以使用,但仍可能随训练能力和配置系统演进调整.
- Internal:内部实现细节,不承诺兼容.用户代码不要直接依赖.

当前项目版本仍是 `0.x`,Stable 表示"本仓库内优先保持兼容",不是已经完成 `1.0` 级别冻结.

### Stable API

#### 配置入口

```python
from xdl.config import setup_from_yaml, TrainSetup, load_structured_dataclass_config
```

承诺:

- `setup_from_yaml(config_path, device=None)` 是 YAML 配置主入口.
- 返回值是 `TrainSetup`.
- `TrainSetup.create_model()` 返回可交给 `Trainer.fit()` 的 `CoreModel` 包装对象.
- `load_structured_dataclass_config(ConfigType, config_path, overrides=None)` 是训练入口可复用的轻量 dataclass 配置加载工具.

#### 训练入口

```python
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
```

承诺:

- `Trainer` 是训练主循环入口.
- `Trainer.fit(model, train_dataloader, val_dataloader=None, inference_data=None)` 是训练主路径.
- `CoreModel` 是用户自定义任务逻辑的基类.
- `TrainSetupModel` 是配置流到 `CoreModel` 的适配层.
- `CoreModel.log(name, value, prefix=None)` 和 `CoreModel.log_metrics(metrics, prefix=None)` 是指标记录入口;`value` 支持 Python 数值或单元素 `torch.Tensor`,内部记录为 `float`;`prefix="train"` 会生成 `train_loss` 这类兼容键名.
- `CoreModel.manual_backward(loss)` 是手动优化的反向传播入口.
- `CoreModel.manual_optimization_step(loss, optimizer=None, model=None, max_grad_norm=None)` 是手动优化和梯度累积的薄模板, 返回当前 micro step 是否执行了 `optimizer.step()`.
- `CoreModel` 的训练步语义属性 `micro_step`,`accumulation_steps`,`micro_step_in_accumulation`,`optimizer_step`,`is_accumulation_start`,`is_accumulation_boundary`,`should_optimizer_step` 可用于手动梯度累积.
- `CoreModel.configure_device_objects()` 和 `CoreModel.on_after_device_setup()` 可用于声明并处理第三方 pipeline 等额外设备迁移对象.
- `Trainer` 暴露同名只读属性,方便 callback 或外层逻辑读取当前 step / accumulation 状态.
- `Trainer.load_checkpoint(model, checkpoint_path, map_location="cpu", format="pt")` 会恢复模型状态和 checkpoint 中的 callback state.

兼容路径:

```python
from xdl.trainer.coreModel import CoreModel
from xdl.trainer.trainer import Trainer
from xdl.trainer.trainSetupModel import TrainSetupModel
```

这些历史路径当前继续可用,但新代码优先使用 `from xdl.trainer import ...`.

#### 回调入口

```python
from xdl.callbacks import Callback
```

承诺:

- `Callback` 是自定义训练生命周期扩展的基类.
- 回调优先级规则保持稳定:数值越小越先执行,默认值由 callback 系统提供.

常用内置回调也可从 `xdl.callbacks` 导入,例如:

```python
from xdl.callbacks import (
    ModelCheckpoint,
    PreviewCallback,
    SaveTrainableStateCallback,
    TqdmCallback,
)
```

内置回调的类名导出会尽量保持兼容;具体构造参数如果调整,应通过文档和测试说明迁移方式.

#### Registry 入口

```python
from xdl.utils.registry import (
    Registry,
    register_model,
    register_dataset,
    register_optimizer,
    register_scheduler,
    register_loss,
    register_metric,
    register_transform,
    register_collate,
)
```

承诺:

- `Registry` 提供注册,查找,列举能力.
- `register_*("Name")(ClassOrFunction)` 是自定义组件接入方式.
- 支持的注册类型保持为:MODEL,DATASET,OPTIMIZER,SCHEDULER,LOSS,METRIC,TRANSFORM,COLLATE.

当前推荐的内置 dataset 主路径包括:

- `ImageFolderDataset`
- `ImageFolderClassificationDataset`
- `ImageTextSidecarDataset`
- `RecordDataset`
- `RecordClassificationDataset`
- `RecordRegressionDataset`
- `RecordMultiLabelClassificationDataset`
- `ImageMaskSidecarDataset`
- `RecordSegmentationDataset`
- `RecordDetectionDataset`
- `RecordImageTextDataset`
- `RecordTextDataset`
- `RecordPairDataset`
- `RecordTripletDataset`
- `ImageEditDataset`

兼容说明: 历史 `Manifest*` dataset 名称和 registry 键仍作为别名保留, 例如 `ManifestRegressionDataset` 等价于 `RecordRegressionDataset`, `ManifestImageEditCollate` 等价于 `ImageEditCollate`. 新代码优先使用 `Record*` / `ImageEdit*` 名称.

#### 常用工具入口

```python
from xdl.utils import resolve_dtype, save_yaml, seed_everything
```

承诺:

- `resolve_dtype(dtype_name, device_name)` 将 `fp32`,`fp16`,`bf16`,`auto` 等常见字符串解析为 `torch.dtype`.
- `seed_everything(seed)` 设置 Python `random` 和 PyTorch 随机种子;CUDA 可用时也会设置 CUDA 随机种子.
- `save_yaml(payload, path)` 将包含 `Path`,`tuple`,`list` 和嵌套 mapping 的配置快照保存为 YAML.

#### 安装后用法入口

```python
import xdl

text = xdl.get_usage_text()
xdl.print_usage()
```

命令行入口:

```bash
python -m xdl.usage
xdl-usage
```

承诺:

- `get_usage_text()` 返回随 wheel 分发的单文件使用说明.
- `print_usage()` 打印同一份说明.

### Provisional API

以下 API 当前可用,但仍处于演进期:

- `xqt` 顶层实验入口: `load_xqt_config`, `run_xqt_recipe`, `preflight_xqt_config`, `XQTConfig`, `ArtifactManifest`, `ArtifactRecord`, `MetricRecord`.
- `xqt` 到 XDL 的适配入口: `xdl_setup_to_xqt_context`, `xdl_checkpoint_to_xqt_context`, `load_checkpoint_into_model`.
- `xdl.config` 中的 schema dataclass,例如 `ConfigSchemaV1`,`RuntimeConfig`,`TrainerConfig`.
- `xdl.config` 中的 resolver 工具,例如 `load_config_with_schema()`,`merge_with_schema()`,`resolve_config()`.
- Accelerate,DeepSpeed,FSDP 相关配置字段和行为.
- 具体内置模型,数据集,loss,metric,optimizer,scheduler 的注册名称和参数细节.
- `RecordDatasetBase` 等 dataset 扩展基类.
- 新增 dataset 模板的字段细节与返回结构, 包括 `ImageFolderDataset`, `ImageTextSidecarDataset`, `ImageMaskSidecarDataset`, `RecordMultiLabelClassificationDataset`, `RecordTextDataset`, `RecordTripletDataset`.
- 复杂 callback 的完整构造参数,例如监控,日志,采样,设备统计类回调.

使用这些 API 时,建议通过测试固定自己的项目契约.

### Internal API

以下内容不承诺兼容:

- 以下划线开头的函数,类,属性和模块级变量.
- `xdl.config.builder`,`xdl.config.setup` 中的内部辅助函数.
- `Trainer`,`CoreModel`,callback 内部状态字段,除公开 property 和文档明确说明的字段外.
- 具体文件布局,例如 `coreModel.py`,`trainer.py` 内部实现函数.
- 临时研究,实验,工具目录中的脚本入口.
- 未列入 Provisional 的 `xqt` 子模块细节,包括 `xqt.core`, `xqt.pipeline`, `xqt.quant`, `xqt.prune`, `xqt.distill`, `xqt.diffusion_distill`, `xqt.export` 内部类和函数.这些接口仍需经过真实 recipe,测试和文档验证后再提升.

如果用户代码必须依赖 Internal API,建议先把需求提升为明确的公共 API,再补文档和测试.

### 废弃策略

对 Stable API 做破坏性变更时,遵守下面流程:

1. 新增推荐路径或新参数.
2. 旧路径继续保留,并在必要时发出 `DeprecationWarning`.
3. 文档写清楚替代方式.
4. 至少保留一个后续版本窗口.
5. 删除旧 API 只能发生在明确的破坏性版本中.

项目进入 `1.0.0` 前,可以保留更快的迭代速度,但 Stable API 仍应优先保持兼容.

### 版本规则

建议按语义化版本管理:

- Patch:bugfix,文档,测试,非破坏性兼容修复.
- Minor:新增公共能力,扩展配置字段,增加组件.
- Major:删除或破坏 Stable API.

当前 `0.x` 阶段仍允许调整设计,但应避免无提示破坏本文档列出的 Stable API.

## XDL HTML 阅读页样式规范

本文是给 Codex 和开发者看的 HTML 样式工作规范. 它不定义框架 API 或训练行为,只约束仓库自有 HTML 阅读页的视觉系统,主题管理和可维护方式.

### 适用范围

必须遵守:

- `docs/html/*.html`
- `research/**/*.html`
- `learn/**/*.html`
- 仓库中新写的面向人类阅读的静态 HTML 页面

不纳入统一改造:

- `third_party/**` 下游项目或生成文档
- 外部工具生成后不打算人工维护的 HTML
- 临时调试导出的 HTML,除非提交为长期阅读页

### 核心原则

- HTML 是给人类阅读的,优先清晰,美观,易扫读,便于交互.
- `docs/md/` 和 AGENTS 是给 agents/开发者看的,优先事实边界,实现约束,索引和检查清单.
- HTML 不作为唯一事实源. 涉及 API,配置,训练行为或目录边界时,先更新 `docs/md/` 中的对应 MD 或源码说明,再同步 HTML.
- 自有 HTML 默认使用统一样式入口 `docs/html/assets/xdl-doc.css`.
- 新增页面不要复制大段内联 `<style>`. 页面可以有少量局部 class,但颜色,间距,字号,圆角,阴影和主题必须来自统一 token.
- 不依赖外部 CDN 做样式. 数学渲染等功能脚本如果已有使用场景可以保留,但视觉系统仍走本地 CSS.

### 统一样式入口

项目级阅读样式入口:

```html
<link rel="stylesheet" href="assets/xdl-doc.css">
<script defer src="assets/xdl-theme.js"></script>
```

跨目录引用时按 HTML 或 CSS 所在路径调整相对路径. 例如研究页已有自己的 `assets/research-doc.css`,应在该 CSS 顶部导入统一样式:

```css
@import "../../../docs/html/assets/xdl-doc.css";
```

允许保留目录专属 CSS,但它应该做两件事:

- 复用统一 token,例如 `--xdl-page`, `--xdl-surface`, `--xdl-text`, `--xdl-accent`, `--xdl-border`, `--xdl-radius`, `--xdl-shadow`.
- 只补充该目录真正需要的组件,例如专题交互图,研究报告发布物轨道或课程独有可视化.

### CSS 收拢边界

`docs/html/assets/xdl-doc.css` 是仓库自有 HTML 阅读页的唯一公共样式入口. 它负责:

- 主题 token,强调色,深浅色和旧变量别名.
- 基础排版,链接,表格,代码块,图片,打印和响应式行为.
- 通用页面组件,例如 `.topbar`, `.hero`, `.button`, `.card`, `.section`, `.layout`, `.toc`, `.article`, `.callout`, `.note`, `.table-wrap`, `.flow`.
- 长文页面如需让正文保持与标题区同宽,同时把目录做成左侧停靠的大组件,优先把这类双栏骨架提升到 `xdl-doc.css`,不要在目录专属 CSS 里重复实现一套侧栏逻辑.
- 通用数学/教程组件,例如 `.formula`, `.math-block`, `.mermaid-wrap`, `.example`, `.practice`, `.checklist`, `.chapter-note`, `.chapter-nav`, `.footer-nav`.
- `body.xdl-style-atlas`, `body.xdl-style-ledger` 两种项目级版式模板,以及 `body.math-doc-page`, `body.research-page` 这类只能叠加在两种模板上的语义桥接.

目录专属 CSS 只允许作为薄入口和局部扩展:

- 第一行必须 `@import` 对应相对路径的 `xdl-doc.css`.
- 文件里只写带目录或页面命名空间的局部组件,例如 `body.lens-report ...` 或 `.flash-attention-page ...`.
- 不复制 reset,主题变量,字体栈,topbar,hero,layout,toc,card,table,figure,callout,formula,practice 或章节导航等公共规则.
- 同一组件被两个以上目录需要时,先提升到 `xdl-doc.css`,再删除目录里的重复实现.
- 不为单页新增一个只改颜色,间距或卡片样式的 CSS 文件. 优先复用已有 body class,统一 token 和公共组件.
- 新增 CSS 文件前先用 `rg --files -g '*.css'` 查现有入口,再用 `rg -n "<style|style=|\\.css"` 查是否已有重复写法.

### 下次编写 HTML 时怎么调用

写新 HTML 前先判断页面放在哪里,再复制对应入口.

`docs/html/*.html`:

```html
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>页面标题</title>
  <link rel="stylesheet" href="assets/xdl-doc.css">
  <script defer src="assets/xdl-theme.js"></script>
</head>
```

`research/<topic>/*.html` 且同目录有 `assets/research-doc.css`:

```html
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>页面标题</title>
  <link rel="stylesheet" href="assets/research-doc.css">
  <script defer src="../../docs/html/assets/xdl-theme.js"></script>
</head>
```

对应 `assets/research-doc.css` 顶部必须导入统一样式:

```css
@import "../../../docs/html/assets/xdl-doc.css";
```

`learn/math/<course>/*.html` 且同目录有课程 CSS:

```html
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>页面标题</title>
  <link rel="stylesheet" href="course.css">
</head>
```

对应课程 CSS 顶部必须导入统一样式:

```css
@import "../../../docs/html/assets/xdl-doc.css";
```

需要页面内主题按钮时使用统一控件:

```html
<div class="theme-switcher" role="group" aria-label="页面主题">
  <button type="button" data-theme-value="system">系统</button>
  <button type="button" data-theme-value="light">日间</button>
  <button type="button" data-theme-value="dark">夜间</button>
  <button type="button" data-theme-value="sepia">暖纸</button>
  <button type="button" data-accent-value="teal">青</button>
  <button type="button" data-accent-value="blue">蓝</button>
</div>
```

页面结构优先使用这个基础骨架:

```html
<body>
  <a class="skip-link" href="#main">跳到正文</a>
  <header class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="index.html">XDL Docs</a>
      <nav aria-label="页面导航">
        <a href="#overview">概览</a>
      </nav>
    </div>
  </header>

  <section class="hero">
    <div>
      <p class="eyebrow">Reading Page</p>
      <h1>页面标题</h1>
      <p class="lead">一句话说明页面给人类读者解决什么问题.</p>
    </div>
  </section>

  <main id="main">
    <section id="overview" class="section">
      <div class="section-head">
        <h2>概览</h2>
        <p>先给读者主线,再进入细节.</p>
      </div>
    </section>
  </main>
</body>
```

### 主题和色彩

统一 CSS 支持:

- `system`: 跟随系统.
- `light`: 日间模式.
- `dark`: 夜间模式.
- `sepia`: 暖纸阅读模式.

统一 CSS 支持强调色:

- `teal`: 默认.
- `blue`
- `violet`
- `amber`
- `rose`
- `green`

主题状态写在根元素:

```html
<html data-theme="dark" data-accent="blue">
```

交互控件使用:

```html
<button type="button" data-theme-value="dark">夜间</button>
<button type="button" data-accent-value="blue">蓝</button>
```

`xdl-theme.js` 会把选择写入 `localStorage`,并给按钮同步 `aria-pressed`. 如果页面不需要显式主题按钮,只引用 CSS 也会自动跟随系统深浅色.

### 设计 token

新增或重构自有 HTML/CSS 时优先使用这些 token:

```css
--xdl-page
--xdl-surface
--xdl-surface-soft
--xdl-surface-muted
--xdl-text
--xdl-muted
--xdl-muted-strong
--xdl-border
--xdl-border-strong
--xdl-accent
--xdl-accent-strong
--xdl-accent-soft
--xdl-code-bg
--xdl-code-border
--xdl-code-text
--xdl-radius
--xdl-shadow
--xdl-content
--xdl-reading
```

兼容旧页时可以继续使用别名:

```css
--bg
--paper
--panel
--surface
--ink
--text
--muted
--line
--accent
--accent-soft
--radius
--shadow
```

新代码优先写 `--xdl-*`,旧 CSS 逐步迁移.

### 默认版式

项目自有长期 HTML 必须且只能使用两种版式之一,都由 `docs/html/assets/xdl-doc.css` 提供.

- `xdl-style-atlas`: 面向入口页,导航页,速查页和交互实验页. 特点是工程控制台式信息密度,明显的模块边界,适合卡片,状态块,路线图和快速跳转.
- `xdl-style-ledger`: 面向长文,教程,调研报告和结构说明页. 特点是纸质手册式阅读节奏,正文宽度更克制,适合段落,表格,公式,引用和边注.

页面通过 body class 选择版式:

```html
<body class="xdl-style-atlas">
```

```html
<body class="xdl-style-ledger">
```

目录专属页面可以叠加语义 class,例如:

```html
<body class="xdl-style-ledger math-doc-page">
<body class="xdl-style-ledger research-page">
<body class="xdl-style-atlas flash-attention-page">
```

`math-doc-page`,`research-page`,`rwkv8-page`,`flash-attention-page` 这类 class 只表达页面语义或局部组件命名空间,不能单独出现在 body 上,也不能在 CSS 中复制成第三套页面主题. 如果一个页面同时不像 atlas 也不像 ledger,先改回两种模板之一,再考虑是否需要把通用组件提升到 `xdl-doc.css`.

长段文本里的重点默认使用四种行内强调:

```html
<strong class="inline-key">注册入口</strong>
<strong class="inline-path">record/manifest 模板</strong>
<strong class="inline-rule">优先复用现有模板</strong>
<span class="inline-warning">不纳入通用说明</span>
```

段落之后需要把结论单独托出时使用:

```html
<div class="emphasis-strip">
  <span><b>Rule</b> Record* / ImageEdit* 命名保持一致.</span>
  <span><b>Boundary</b> hair 系列特殊数据集不进入通用说明.</span>
</div>
```

### 版式要求

- 阅读主宽度控制在 `--xdl-reading` 到 `--xdl-content` 之间. 长正文不要铺满超宽屏.
- 页面要有明确的 `header`,正文 `main`,必要时有 sticky `topbar` 或 `toc`.
- 需要可折叠目录时,优先使用 `details.toc > summary + nav.toc-list` 结构,并在公共 CSS 中定义展开/收回两种状态,保证标题区与正文共用同一右侧列宽.
- 标题层级保持真实结构,不要为了变大而跳级.
- 大段正文使用舒展行高,保持短段落,表格必须可横向滚动.
- 卡片只用于并列信息块,工具面板,索引项和局部容器. 不要把整个页面堆成卡片套卡片.
- 常用组件优先复用: `.topbar`, `.hero`, `.button`, `.card`, `.section`, `.toc`, `.route`, `.callout`, `.note`, `.table-wrap`, `.flow`.
- 圆角默认 `8px`. 不要随意做过大的圆角.
- 不使用单一色相铺满全页. 默认中性底色 + 一个强调色 + 少量语义色.
- 交互元素必须有 hover/focus 可见状态,按钮文字要短,不挤压.

### 可访问性和维护

- 页面必须保留 `<meta name="viewport" content="width=device-width, initial-scale=1">`.
- 长页建议加 `.skip-link` 和目录.
- 图片和图示要有 `alt` 或 `aria-label`.
- 不要用颜色作为唯一信息来源. 重要状态同时用标题,标签或文本说明.
- 主题切换后仍要检查代码块,表格,按钮和 callout 的对比度.
- 打印时隐藏导航,目录和主题按钮,正文保持可读.

### 新增页面检查清单

- 已引用 `xdl-doc.css`,没有复制整套内联样式.
- body 已且仅已选择 `xdl-style-atlas` 或 `xdl-style-ledger` 之一.
- 没有 `<style>` 块和散落的 `style=`.
- 需要主题按钮时已引用 `xdl-theme.js`.
- 使用统一 token,没有大面积硬编码颜色.
- 在浅色,深色和窄屏下结构不重叠.
- 表格和公式块可横向滚动.
- HTML 内容和对应 MD/源码事实一致.

## XDL 模块功能边界速查

本文档只做"模块职责速查",不重复长篇架构介绍.详细背景请看 [XDL.md](XDL.md),配置细节请看 [CONFIG.md](CONFIG.md),公共 API 边界请看 [API.md](API.md).

### 1. 总览

```text
xdl/
  callbacks/   训练生命周期扩展
  config/      YAML 到 TrainSetup 的构建链
  dataset/     数据集,transform,collate
  loss/        损失函数
  metric/      评估指标
  model/       模型与工厂函数
  optimizer/   优化器
  scheduler/   学习率调度器
  trainer/     CoreModel / Trainer / TrainSetupModel
  utils/       registry,checkpoint,通用工具
```

### 2. `xdl/callbacks`

职责:

- 日志,进度条,检查点,早停,监控等横切逻辑
- 跟随训练生命周期执行

不负责:

- 具体模型训练逻辑
- 优化器步进
- 数据加载

关键点:

- 优先级数值越小越先执行
- 回调更适合作为观察者,不应承载主训练逻辑

### 3. `xdl/config`

职责:

- 解析 YAML
- 套用 schema v1
- 构建 model / dataset / dataloader / optimizer / scheduler / loss / metrics
- 返回 `TrainSetup`

不负责:

- 替代 `Trainer.fit()`
- 承担模型特化训练逻辑

关键文件:

- `schema.py`
- `resolver.py`
- `builder.py`
- `setup.py`

### 4. `xdl/dataset`

职责:

- 数据集定义
- transform / collate 相关能力
- 通过 registry 对外暴露

不负责:

- 训练循环
- 模型前向

关键点:

- 常见容器 batch 会由 Trainer 递归迁移;第三方自定义对象需要自行处理设备
- 可选依赖应优雅降级

### 5. `xdl/loss`

职责:

- 提供单个或组合损失函数
- 通过 registry 接入配置系统

关键点:

- 新增 loss 后要在 `__init__.py` 中集中注册
- 多 loss 配置会由 `build_loss()` 构建成 `WeightedLoss`

### 6. `xdl/metric`

职责:

- 提供训练与验证指标
- 按配置构建指标列表

关键点:

- 指标通常需要稳定的 `update / compute` 或可调用语义
- 注册名与导出要同步维护

### 7. `xdl/model`

职责:

- 提供模型类与工厂函数
- 通过 registry 暴露给纯代码路径和 YAML 路径

不负责:

- 训练循环编排
- 日志和检查点

关键点:

- 当前源码中包含分类,ViT,生成,分割和底层超分模型
- 新模型应在 `__init__.py` 中集中注册

### 8. `xdl/optimizer`

职责:

- 自定义优化器封装
- 与 PyTorch optimizer 一起统一进入构建链

关键点:

- `build_optimizer()` 支持 `target_modules` 和 `param_groups`
- 需要明确参数选择范围和状态初始化语义

### 9. `xdl/scheduler`

职责:

- 学习率调度器与工厂函数
- 为训练流程提供统一调度入口

关键点:

- 要区分 step 级还是 epoch 级调用语义
- 配置侧由 `build_scheduler()` 负责实例化

### 10. `xdl/trainer`

职责:

- `CoreModel`:任务逻辑抽象
- `Trainer`:训练循环编排
- `TrainSetupModel`:把配置流构建的外部组件桥接到 `CoreModel`

关键点:

- 稳定公共入口是 `from xdl.trainer import CoreModel, Trainer, TrainSetupModel`
- `Trainer.fit()` 先调用 `model.setup("fit")`
- `CoreModel.training_step()` 是手动优化模式
- 手动累积优先用 `micro_step` / `is_accumulation_boundary` 等公开 helper
- 指标记录可用 `self.log("loss", value, prefix="train")` 生成 `train_loss`
- callback 在这里被统一调度

### 11. `xdl/utils`

职责:

- registry
- checkpoint
- tiling
- 权重与通用辅助函数

关键点:

- 这是基础设施层,改动影响面大
- registry 只做名字到对象的映射,不掺配置解析
- 自定义组件稳定接入方式是 `register_*("Name")(ClassOrFunction)`

### 12. 依赖关系

可以把依赖方向简化理解成:

```text
model / dataset / loss / metric / optimizer / scheduler
        ↓
      utils.registry
        ↓
      config.builder / setup
        ↓
       trainer
        ↓
     callbacks
```

更准确地说:

- 组件层依赖 registry 暴露自己
- config 依赖 registry 构建组件
- trainer 依赖 config 产物或纯代码组件
- callbacks 依赖 trainer 生命周期

## XDL 当前优化方向

本文档只保留当前仍然有效的改进项,不重复过时问题,也不重新解释模块结构.

### P0 - 低风险立即可做

#### 1. `xdl/model/AGENTS.md` 与当前代码继续对齐

原因:

- `xdl/model/` 已包含分类,ViT,生成,分割,低层超分等多类条目
- 模块文档应该持续跟上注册表真实状态

建议:

- 保持已注册条目清单准确
- 简化示例,避免文档再次膨胀成架构分析

#### 2. 训练入口文档补最短主路径示例

原因:

- 当前最真实的接入方式其实是两条:纯代码路径和 `setup_from_yaml()` 路径
- 文档已说明方向,但还可以继续压缩成更短的复制即用示例

建议:

- 在 `README.md` 或 `docs/md/XDL.md` 中只保留一段最短 runnable 代码

### P1 - 质量提升

#### 3. `xdl/dataset/hair/` 的毛发数据集增强逻辑去重

原因:

- `xdl/dataset/hair/` 下的毛发数据集仍可能存在共享增强逻辑

建议:

- 提取共享 transform 工厂
- 保持数据集文件关注自己的索引与标注逻辑

#### 4. tests 覆盖继续向 trainer / callbacks / optimizer 扩展

原因:

- 当前配置相关链路相对清楚
- 训练编排,回调组合和优化器行为更值得补回归保护

建议:

- `trainer` 主路径
- callback 组合行为
- 自定义 optimizer 最小 step 行为

#### 5. 文档与目录局部说明继续标准化

原因:

- 这轮已经给大部分目录补了 `AGENTS.md` / `CLAUDE.md`
- 还可以再统一章节顺序和粒度

建议:

- `职责`
- `当前内容`
- `修改约束`
- `验证建议`

### P2 - 中期增强

#### 6. 配置系统继续补强 dataset / collate / task 级抽象

原因:

- 当前 config 系统已经能稳定构建组件
- 但 task 级差异仍主要留在训练脚本或 `CoreModel` 子类里

建议:

- 只在确有复用价值时上提抽象
- 避免把任务私有逻辑硬塞进 schema

#### 7. `xdl/dataset/` 按数据形态继续补模板

原因:

- 当前 manifest 主路径已经基本成形
- `ImageFolderDataset` 和 `ImageTextSidecarDataset` 已覆盖纯图片目录与 `image + txt` 基础入口
- `ImageMaskSidecarDataset` 已覆盖常见 `images/000.png` 对应 `masks/000.png` 的分割入口
- 实际项目里经常先面对"数据怎么摆",再决定任务类型

建议把 dataset 继续按两层补齐:

1. 样本语义形态

- `image + label`
- `image + target`
- `image + labels`
- `image + text`
- `text (+ target_text)`
- `pair`
- `triplet`
- `image + mask`
- `image + boxes + labels`
- `image edit`

2. 磁盘组织形态

- manifest 显式字段
- 目录分类
- basename sidecar 对齐
- 纯图片目录
- 标准格式标注(COCO / YOLO / VOC / keypoint)

建议优先级:

1. 更通用的 `BasenameAlignedDataset`
   - 复用当前 sidecar helper
   - 覆盖 `image + json` / `image + label` 这类 sidecar 结构

2. 标准格式适配
   - `COCODetectionDataset`
   - `COCOSegmentationDataset`
   - keypoint 模板

落地原则:

- manifest 继续作为长期推荐主路径
- sidecar / image-folder 模板作为低门槛接入路径
- 不把所有数据组织方式都硬塞成一个超大 dataset 类
- 优先沉淀成可注册,可 YAML 构建,可测试的通用模板,而不是只在训练脚本里临时实现

#### 8. 分布式与大模型接入路径继续沉淀

原因:

- 当前 Trainer 已支持 accelerate 方向
- 但更复杂的大模型,外部 pipeline,LoRA 保存路径仍偏工程化

建议:

- 优先在真实训练入口中沉淀稳定模式
- 成熟后再回抽到 callback 或 config 层
