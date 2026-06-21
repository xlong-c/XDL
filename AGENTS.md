# XDL — 模块化深度学习框架

PyTorch 深度学习框架, 组件注册系统 + 回调生命周期. Python 3.12+, PyTorch 1.12+, CUDA 可选.

## 行为准则

- **语言**：所有推理和回答均使用中文
- 做非 trivial 改动前，先用 `grep` / `find` / Agent 搜索代码库，理解现有模式再动手
- 多文件重构或架构级变更时，先进入规划模式（EnterPlanMode）制定方案
- 不确定文件位置或调用关系时，优先搜索而非猜测
- 涉及第三方库用法、API 变更、最佳实践等外部知识时，先用 WebSearch 查最新文档
- **脚本参数**：Python 脚本不要使用命令行参数解析库，优先使用 YAML 配置或代码内显式配置
- **配置规范**：新增或重构 YAML/层级配置解析时，优先使用 `OmegaConf` 统一处理加载、合并、插值、resolver 和 `DictConfig`/`ListConfig` 到普通容器的转换；不要在训练入口或工具脚本里散落 `yaml.safe_load` + 手写合并逻辑，除非只是兼容旧路径或读写极小的固定结构文件
- **结构化配置加载规则**: 训练入口或研究脚本里的轻量 `load_config` 优先遵循 `dataclass/structured config` 定义默认值和 schema, 再由 YAML 直接覆盖的顺序. 除非有明确兼容需求, 不要在 `load_config` 中额外做路径重写, 字符串 `"null"` 兼容, 旧字段迁移, clamp/奇偶修正, 或 list/tuple 强转. 需要约束时优先让 schema/OmegaConf 报错, 或在业务使用处显式校验
- **半角符号**: 文档或者注释之类使用点,括号,引号,冒号等一律写半角,不要混入全角符号. 写作后可用 `XDL_PUNCT_PATHS=<path> python scripts/normalize_punctuation.py` 自动归一,用 `XDL_PUNCT_CHECK=1` 只检查不写入
- **文档第一规则**: `docs/md/` 是给 agents 和开发者写代码前看的工作文档,`docs/html/` 是给人类用户阅读的可视化文档. 新增长期 MD 放 `docs/md/`,新增自有 HTML 放 `docs/html/`.
- **HTML 阅读页样式**: 仓库自有 HTML 是给人类阅读的可视化层,新增或重构 `docs/html/`,`research/`,`learn/` 等目录下的 HTML/CSS 时,先遵循 [docs/md/README.md#xdl-html-阅读页样式规范](docs/md/README.md#xdl-html-阅读页样式规范). 项目自有长期 HTML 必须且只能归入 `xdl-style-atlas` 或 `xdl-style-ledger` 两种 body 模板之一;`math-doc-page`,`research-page`,`flash-attention-page` 等只能作为语义叠加 class,不能成为第三套视觉模板. 默认复用 `docs/html/assets/xdl-doc.css` 的主题 token 和公共组件,不要复制大段内联 `<style>`,不要使用散落的 `style=`,不要复制公共版式或为单页另写一套主题系统;需要交互式主题切换时复用 `docs/html/assets/xdl-theme.js`.
- **知识图谱更新**：完成大的代码改动（新增/删除模块、重命名公开符号、模块间调用关系变化等结构性变更）后，手动运行 `index_repository` 刷新知识图谱，保持图与代码一致。

## 架构

```
xdl/
├── callbacks/   # 训练生命周期钩子（优先级越小越先执行，默认 999）
├── dataset/     # 数据集定义
├── loss/        # 损失函数
├── metric/      # 评估指标
├── model/       # 模型架构（CNN、ViT、生成模型等）
├── optimizer/   # 优化器封装
├── scheduler/   # 学习率调度器
├── trainer/     # 训练器核心
└── utils/       # 注册系统（registry）
```

其他目录：`config/`（YAML 配置）、`learn/`（CUDA 内核实验）、`tools/`（数据工具）、`tests/`。

## 数据集存放约定

- **所有数据集和样本数据一律放进 `data/`**，不要新建顶层 `datasets/`、`raw/`、`resources/` 等平行目录。历史上仓库根目录曾有 `datasets/`（存放 `coco8` 样本），现已并入 `data/`，统一归 `data/` 管理。
- `data/` 被 `.gitignore` 整体忽略，数据本体（图片、标注、压缩包、大文件）一律不入库；只有 `data/AGENTS.md` 通过例外规则保留，作为目录说明。需要样本数据时用下载脚本（如 `tools/dataset/`）在本地准备，不要把数据本体 commit 进仓库。
- 详细规则见 [data/AGENTS.md](data/AGENTS.md)。

## 核心约束

- **所有组件必须通过注册系统注册**：在对应 `__init__.py` 中用 `register_*("Name")(Class)` 集中注册
- 支持八种注册类型：MODEL, DATASET, OPTIMIZER, SCHEDULER, LOSS, METRIC, TRANSFORM, COLLATE
- **回调优先级**：数值越小越先执行，未指定时默认 999
- 所有函数必须加类型注解

## 三种使用方式

1. **纯代码**（VAE/GAN 示例）：手动实例化所有组件，`python train_VAE.py`
2. **YAML 配置（推荐）**：`setup = setup_from_yaml('config/xxx.yaml')` → 返回 `TrainSetup` dataclass，一行拿到 model/optimizer/train_loader 等全部组件
3. **DeepSpeed 分布式**：`deepspeed train_script.py --deepspeed config/deepspeed_config.json`

## 核心文档入口

以下文档是 XDL 的核心规范，不是普通参考链接。`docs/md/README.md` 现在是 XDL MD 正文事实源, XDL 安装,框架,配置,dataset,API,HTML 样式,模块边界和优化方向都在这个文件的对应章节维护。改动框架源码、训练入口、配置系统、公开 API、打包安装、文档索引或模块边界前，必须先阅读与改动范围对应的章节，并遵循其中的约束；如果实现行为发生变化，必须同步更新相关章节。阶段性研究资料仍放在 `research/`，不能替代这些长期文档。

必读规则：

- 改动公开 API、导出符号、兼容策略或 Stable / Provisional / Internal 边界时，必须先读并遵循 [docs/md/README.md#xdl-api-稳定边界](docs/md/README.md#xdl-api-稳定边界)，行为变化必须同步更新它。
- 改动训练生命周期、`CoreModel`、`Trainer`、回调调用顺序、手动优化、batch 迁移或日志行为时，必须先读并遵循 [docs/md/README.md#xdl-项目结构与使用说明](docs/md/README.md#xdl-项目结构与使用说明) 和 [xdl/trainer/README.md](xdl/trainer/README.md)。
- 改动 YAML 配置、schema、`target + params` 组织方式或 `setup_from_yaml` 构建行为时，必须先读并遵循 [docs/md/README.md#xdl-config-系统说明](docs/md/README.md#xdl-config-系统说明)。
- 改动安装、依赖、wheel 分发、可运行入口或环境验证时，必须先读并遵循 [docs/md/README.md#xdl-安装与验证](docs/md/README.md#xdl-安装与验证) 和 [xdl/USAGE.md](xdl/USAGE.md)。
- 改动子模块职责、目录边界或把逻辑在 `xdl/`、`tools/`、`examples/`、`train/` 等目录之间迁移时，必须先读并遵循 [docs/md/README.md#xdl-模块功能边界速查](docs/md/README.md#xdl-模块功能边界速查)。
- 改动文档结构、文档索引或长期文档边界时，必须先读并遵循 [docs/AGENTS.md](docs/AGENTS.md) 和 [docs/md/README.md](docs/md/README.md)。
- 改动自有 HTML 阅读页视觉系统,主题 token,公共 CSS 或主题切换交互时,必须先读并遵循 [docs/md/README.md#xdl-html-阅读页样式规范](docs/md/README.md#xdl-html-阅读页样式规范).

- [docs/md/README.md](docs/md/README.md) — XDL MD 正文事实源,推荐阅读顺序和各主题边界
- [docs/md/README.md#xdl-安装与验证](docs/md/README.md#xdl-安装与验证) — 安装、环境验证和当前可运行入口
- [docs/md/README.md#xdl-项目结构与使用说明](docs/md/README.md#xdl-项目结构与使用说明) — 框架定位、核心分层、训练入口、`CoreModel` / `Trainer` 生命周期
- [xdl/USAGE.md](xdl/USAGE.md) — 随 wheel 分发的单文件用法摘要，安装后可通过 `xdl-usage` 查看
- [docs/md/README.md#xdl-config-系统说明](docs/md/README.md#xdl-config-系统说明) — YAML 配置系统、schema v1、`target + params` 组织方式
- [docs/md/README.md#xdl-api-稳定边界](docs/md/README.md#xdl-api-稳定边界) — Stable / Provisional / Internal API 边界和兼容策略
- [docs/md/README.md#xdl-html-阅读页样式规范](docs/md/README.md#xdl-html-阅读页样式规范) - 自有 HTML 阅读页统一样式,主题 token,色彩,交互规范和 CSS 收拢边界
- [docs/md/README.md#xdl-模块功能边界速查](docs/md/README.md#xdl-模块功能边界速查) — 各源码子模块职责速查与边界
- [docs/md/README.md#xdl-当前优化方向](docs/md/README.md#xdl-当前优化方向) — 当前仍有效的框架后续优化方向
- [xdl/trainer/README.md](xdl/trainer/README.md) — 训练器子模块说明，含手动优化、梯度累积 helper、日志命名和 batch 迁移行为

## XDL 训练脚本接入备忘

编写新的训练入口时，优先先看 `train_VAE.py`、`train_TwinFlow.py`、`examples/*finetune.py` 和 `xdl/trainer/{trainer.py,coreModel.py}`。这几个文件能最快说明 XDL 的真实生命周期。

- `Trainer.fit()` 会先调用 `model.setup("fit")`，再执行 Trainer 的设备/优化器 setup；大模型、diffusers pipeline、PEFT LoRA 等重组件适合在 `CoreModel.setup()` 中懒加载。
- `CoreModel.training_step()` 是**手动优化模式**，Trainer 不会自动 `zero_grad/backward/step`。训练步里需要自行调用 `optimizer.zero_grad()`、`self.manual_backward(loss)`、`self.clip_gradients(...)`、`optimizer.step()`，并用 `self.log()` 记录指标；新代码可用 `self.log("loss", loss, prefix="train")` 生成 `train_loss`。
- `configure_optimizers()` 在 `setup()` 之后由 Trainer 调用；如果优化器依赖懒加载出来的模块，必须确保这些模块已经在 `setup()` 中初始化。
- 标准非 Accelerate 路径下，Trainer 只会把 `CoreModel.__dict__` 中的 `nn.Module` 属性迁移到设备；`diffusers.Pipeline` 不是 `nn.Module`，需要把底层 `vae/text_encoder/transformer` 注册成模块属性，或在模型钩子中显式 `pipe.to(device)`。
- Trainer 对 batch 的自动设备迁移已递归支持 `Tensor / dict / list / tuple / dataclass`；第三方自定义对象仍建议在 `training_step()` 内显式 `.to(self.device)`。
- `on_train_step_start()` 会在 `training_step()` 前递增 `_total_train_steps`；新代码优先使用 `micro_step`、`accumulation_steps`、`is_accumulation_start`、`is_accumulation_boundary`、`should_optimizer_step` 等 helper。Trainer 构造参数里的 `gradient_accumulation_steps` 不会替代手动优化逻辑。
- 自定义保存优先用 Callback。XDL 的通用 `ModelCheckpoint` 会保存 `CoreModel` 的 state_dict/optimizer 状态；diffusers/PEFT LoRA 这类权重通常要写专门的 callback 调 `save_pretrained()` 或 `StableDiffusion3Pipeline.save_lora_weights()`。
- `inference_data` 会走验证/推理周期并调用 `CoreModel.inference(data)`，适合生成式模型的采样预览；没有 val loader 时也可以只传 prompts 做周期性采样。
- 对外部 `third_party/` 代码不要只信 README 路径，先用 `grep`/`find` 查真实文件；如果第三方目录不是 Python package，可在脚本里用受控的 `sys.path.insert()` 或 `importlib.util.spec_from_file_location()` 加载。
- 训练脚本参数继续遵守本仓库约束：不要引入命令行参数解析库，优先 YAML 配置；复杂配置加载、合并和插值优先走 `OmegaConf` 或 `xdl.config` 主链路，需要切换配置时可用环境变量指向 YAML。

## 命令

```bash
pip install -e ".[all]"          # 安装
python train_VAE.py              # VAE 训练 (MNIST)
python train_GAN.py              # GAN 训练 (MNIST)
CUDA_VISIBLE_DEVICES=0,1 python train_VAE.py  # 指定 GPU
pytest tests/ -v --cov=xdl       # 测试
```

## 注意事项

- GPU 内存紧张时用梯度累积、AMP、激活检查点
- DeepSpeed/Accelerate 分布式训练需同步保存检查点
- `learn/` 目录下为 C++23 CUDA 内核实验代码，不遵循 Python 规范

## 子模块文档

### 顶层目录

- [config/AGENTS.md](config/AGENTS.md) — 训练与运行配置目录
- [data/AGENTS.md](data/AGENTS.md) — 本地数据目录（所有数据集统一放 `data/`，不新建 `datasets/` 等平行目录，数据本体不入库）
- [docs/AGENTS.md](docs/AGENTS.md) — 仓库文档目录
- [downloads/AGENTS.md](downloads/AGENTS.md) — 下载产物目录
- [examples/AGENTS.md](examples/AGENTS.md) — 示例脚本目录
- [infer/AGENTS.md](infer/AGENTS.md) — 推理脚本目录
- [learn/AGENTS.md](learn/AGENTS.md) — 学习与实验目录
- [others/AGENTS.md](others/AGENTS.md) — 其他资产目录
- [research/AGENTS.md](research/AGENTS.md) — 研究资料目录
- [scripts/AGENTS.md](scripts/AGENTS.md) — 仓库维护脚本目录
- [tests/AGENTS.md](tests/AGENTS.md) — 测试目录
- [third_party/AGENTS.md](third_party/AGENTS.md) — 第三方代码与资产目录
- [tools/AGENTS.md](tools/AGENTS.md) — 数据与工程工具目录
- [train/AGENTS.md](train/AGENTS.md) — 训练入口目录
- [xdl/AGENTS.md](xdl/AGENTS.md) — XDL 框架源码根目录
- [xqt/AGENTS.md](xqt/AGENTS.md) — 量化与实验脚本目录

### XDL 核心子模块

- [xdl/callbacks/AGENTS.md](xdl/callbacks/AGENTS.md) — 回调系统
- [xdl/config/AGENTS.md](xdl/config/AGENTS.md) — 配置构建子模块
- [xdl/dataset/AGENTS.md](xdl/dataset/AGENTS.md) — 数据集子模块
- [xdl/loss/AGENTS.md](xdl/loss/AGENTS.md) — 损失函数子模块
- [xdl/metric/AGENTS.md](xdl/metric/AGENTS.md) — 评估指标子模块
- [xdl/model/AGENTS.md](xdl/model/AGENTS.md) — 模型架构子模块
- [xdl/optimizer/AGENTS.md](xdl/optimizer/AGENTS.md) — 优化器子模块
- [xdl/scheduler/AGENTS.md](xdl/scheduler/AGENTS.md) — 学习率调度器子模块
- [xdl/trainer/AGENTS.md](xdl/trainer/AGENTS.md) — 训练器核心子模块
- [xdl/utils/AGENTS.md](xdl/utils/AGENTS.md) — 注册系统与工具子模块
