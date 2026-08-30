# XDL Workspace - 工作区规范

本工作区包含两个项目:

- **XDL** - 模块化深度学习框架 (PyTorch, 组件注册 + 回调生命周期). 训练侧.
- **XQT** - 模型压缩与部署目录 (图变换, 量化, 剪枝, 导出, benchmark). 部署侧.

XDL 负责训练, XQT 只负责模型本身, 二者通过 checkpoint / 模型产物衔接, 不互相接管职责.

---

## Python 之禅

> The Zen of Python, by Tim Peters
 1. 优美胜于丑陋（Beautiful is better than ugly.）
 2. 明了胜于晦涩（Explicit is better than implicit.）
 3. 简洁胜于复杂（Simple is better than complex.）
 4. 复杂胜于凌乱（Complex is better than complicated.）
 5. 扁平胜于嵌套（Flat is better than nested.）
 6. 稀疏胜于密集（Sparse is better than dense.）
 7. 可读性很重要（Readability counts.）
 8. 特例不足以特殊到打破规则（Special cases aren't special enough to break the rules.）
 9. 虽然，实用性胜过纯粹性（Although practicality beats purity.）
10. 错误不应该被悄悄传递（Errors should never pass silently.）
11. 除非显式沉默（Unless explicitly silenced.）
12. 面对歧义，拒绝猜测的诱惑（In the face of ambiguity, refuse the temptation to guess.）
13. 应该有一种——最好只有一种——明显的方式做一件事（There should be one--and preferably only one--obvious way to do it.）
14. 虽然，这种方式可能一开始并不明显，除非你是荷兰人（Although that way may not be obvious at first unless you're Dutch.）
15. 现在做胜过不做（Now is better than never.）
16. 但永远不做通常好过 仓促 去做（Although never is often better than right now.）
17. 如果实现难以解释，那就是个坏主意（If the implementation is hard to explain, it's a bad idea.）
18. 如果实现易于解释，那可能是个好主意（If the implementation is easy to explain, it may be a good idea.）
19. 命名空间是个绝妙的主意——让我们多做些这样的东西（Namespaces are one honking great idea -- let's do more of those!）

落到本工作区的几条原则:

- 一种事只有一个明显的做法. 配置走 OmegaConf, 注册走 registry, 训练走 CoreModel + Trainer, 压缩走 XQT session/recipe, 不要另起平行链路.
- 显式优于隐式. 路径重写, 字符串兼容, 字段迁移, 强转这类隐式修正尽量不做, 让 schema/OmegaConf 直接报错, 或在使用处显式校验.
- 实现说不清楚就是坏主意. 新能力必须服务项目本职 (XDL 服务训练, XQT 服务模型本身), 不要塞与本职无关的循环, provider 或 registry.
- Special cases 不够特殊到值得破例. 先搜代码理解既有模式, 不猜路径, 不猜调用关系.

---

## 行为准则

- **语言**: 所有推理和回答均使用中文.
- **默认执行**: 用户目标、修改范围和验收标准已足够明确时, 直接执行; 仅在需求存在实质歧义, 或缺少完成任务所必需的信息时, 再先对齐.
- **用户优先**: 若用户明确说 "先分析"、"先计划"、"先复述"、"不要猜"、"等我确认后再改", 必须严格遵守.
- 做 non-trivial 改动前, 先用 `grep` / `find` / Agent 搜索代码库, 理解既有模式再动手.
- 多文件重构或架构级变更时, 先进入规划模式 (EnterPlanMode) 制定方案.
- 不确定文件位置或调用关系时, 优先搜索而非猜测.
- 涉及第三方库用法, API 变更, 最佳实践等外部知识时, 先用 WebSearch 查最新文档.
- **半角符号**: 文档或注释使用点, 括号, 引号, 冒号等一律写半角, 不混入全角. 写作后用 `XDL_PUNCT_PATHS=<path> python scripts/normalize_punctuation.py` 自动归一, `XDL_PUNCT_CHECK=1` 只检查.
- **知识图谱**: 完成结构性变更 (新增/删除模块, 重命名公开符号, 调用关系变化) 后, 手动运行 `index_repository` 刷新知识图谱. 本工作区在 `codebase-memory-mcp` 中项目名固定为 `root-workspace-xdl`, 调用 `index_status` / `search_graph` / `trace_path` / `get_code_snippet` / `query_graph` 时 `project` 统一传 `root-workspace-xdl`.
- **技能目录**: 仓库 `skills/<name>/SKILL.md` 是项目技能的唯一实体来源. `.claude/skills/<name>` 和 `.codex/skills/<name>` 用相对软链接 (`../../skills/<name>`), Codex 的 `~/.codex/skills/<name>` 用绝对软链接指向仓库内同一目录. 不在工具目录下直接放或复制项目技能本体; 工具自带或第三方技能按其自身安装方式管理.

## 脚本与配置规范

- **脚本参数**: Python 脚本不要使用命令行参数解析库, 优先 YAML 配置或代码内显式配置.
- **配置加载**: 新增或重构 YAML/层级配置解析时, 优先用 `OmegaConf` 统一处理加载, 合并, 插值, resolver 和 `DictConfig`/`ListConfig` 到普通容器的转换; 不要在训练入口或工具脚本里散落 `yaml.safe_load` + 手写合并逻辑, 除非只是兼容旧路径或读写极小固定结构.
- **结构化配置**: 训练入口或研究脚本里的轻量 `load_config` 优先遵循 `dataclass/structured config` 定义默认值和 schema, 再由 YAML 直接覆盖. 除非有明确兼容需求, 不在 `load_config` 里做路径重写, 字符串 `"null"` 兼容, 旧字段迁移, clamp/奇偶修正或 list/tuple 强转; 需要约束时让 schema/OmegaConf 报错, 或在使用处显式校验.
- **类型注解**: 所有函数必须加类型注解.

## 文档规范

- **第一规则**: `docs/md/` 是给 agents 和开发者写代码前看的工作文档, `docs/html/` 是给人类阅读的可视化文档. 新增长期 MD 放 `docs/md/`, 新增自有 HTML 放 `docs/html/`. 阶段性研究资料放 `research/`, 不能替代长期文档.
- **HTML 阅读页样式**: 新增或重构 `docs/html/`, `research/`, `learn/` 等目录下 HTML/CSS 时, 先遵循 [docs/md/README.md#xdl-html-阅读页样式规范](docs/md/README.md#xdl-html-阅读页样式规范). 自有长期 HTML 必须且只能归入 `xdl-style-atlas` 或 `xdl-style-ledger` 两种 body 模板; `math-doc-page`, `research-page`, `flash-attention-page` 等只能作语义叠加 class. 默认复用 `docs/html/assets/xdl-doc.css` 主题 token 和公共组件, 不复制大段内联 `<style>`, 不用散落 `style=`, 主题切换复用 `docs/html/assets/xdl-theme.js`.

---

## XDL 约定

### 组件注册

- 所有组件必须通过注册系统注册, 在对应 `__init__.py` 用 `register_*("Name")(Class)` 集中注册.
- 八种注册类型: MODEL, DATASET, OPTIMIZER, SCHEDULER, LOSS, METRIC, TRANSFORM, COLLATE.
- 回调优先级: 数值越小越先执行, 未指定时默认 999.

### 数据集存放

- 所有数据集和样本数据一律放进 `data/`, 不新建顶层 `datasets/`, `raw/`, `resources/` 等平行目录.
- `data/` 被 `.gitignore` 整体忽略, 数据本体 (图片, 标注, 压缩包, 大文件) 一律不入库; 只有 `data/AGENTS.md` 例外保留. 需要样本数据用下载脚本本地准备.
- 详细规则见 [data/AGENTS.md](data/AGENTS.md).

### 预训练与后训练

- 训练入口按阶段划分: 预训练 (从零训练) 放 `train/pretrain/`, 后训练 (SFT/LoRA 微调, 偏好优化与 RL, 蒸馏, SFT checkpoint 合并) 放 `train/posttrain/`; 划分规则见 [train/AGENTS.md](train/AGENTS.md).
- 框架内后训练组件收拢在 `xdl/post_training/` (偏好优化与蒸馏 loss, rollout/参考模型/SFT 合并/adapter 保存回调); 从零训练用的通用损失仍在 `xdl/loss/`.
- `examples/` 只放示例与演示脚本, 不承载正式训练入口.

### 训练脚本接入

编写训练入口前, 优先先看 `train/pretrain/train_VAE.py`, `train/pretrain/train_TwinFlow.py`, `train/posttrain/train_GRPO.py`, `train/posttrain/*finetune*.py` 和 `xdl/trainer/{trainer.py,core_model.py}`. 关键事实:

- `Trainer.fit()` 先调 `model.setup("fit")`, 再做设备/优化器 setup; 重组件 (大模型, diffusers pipeline, PEFT LoRA) 适合在 `CoreModel.setup()` 中懒加载.
- `CoreModel.training_step()` 是**手动优化模式**, Trainer 不自动 `zero_grad/backward/step`; 需自行调用 `optimizer.zero_grad()`, `self.manual_backward(loss)`, `self.clip_gradients(...)`, `optimizer.step()`, 并用 `self.log()` 记录指标 (新代码用 `self.log("loss", loss, prefix="train")`).
- `configure_optimizers()` 在 `setup()` 之后由 Trainer 调用; 若优化器依赖懒加载模块, 必须确保这些模块已在 `setup()` 中初始化.
- 标准 (非 Accelerate) 路径下 Trainer 只迁移 `CoreModel.__dict__` 中的 `nn.Module` 属性; `diffusers.Pipeline` 非 `nn.Module`, 需把底层 `vae/text_encoder/transformer` 注册成模块属性或显式 `pipe.to(device)`.
- batch 自动设备迁移递归支持 `Tensor / dict / list / tuple / dataclass`; 第三方自定义对象仍建议在 `training_step()` 内显式 `.to(self.device)`.
- 梯度累积优先用 `micro_step`, `accumulation_steps`, `is_accumulation_start`, `is_accumulation_boundary`, `should_optimizer_step` 等 helper; `gradient_accumulation_steps` 构造参数不替代手动优化逻辑.
- 自定义保存优先用 Callback. 通用 `ModelCheckpoint` 存 state_dict/optimizer 状态; diffusers/PEFT LoRA 需专门 callback 调 `save_pretrained()` / `StableDiffusion3Pipeline.save_lora_weights()`.
- `inference_data` 走验证/推理周期调 `CoreModel.inference(data)`, 适合生成模型采样预览.
- 外部 `third_party/` 代码不要只信 README 路径, 先 `grep`/`find` 查真实文件; 非 Python package 时用受控 `sys.path.insert()` 或 `importlib.util.spec_from_file_location()` 加载.

### XDL 注意事项

- GPU 内存紧张时用梯度累积, AMP, 激活检查点.
- DeepSpeed/Accelerate 分布式训练需同步保存检查点.
- `learn/` 下为 C++23 CUDA 内核实验代码, 不遵循 Python 规范.

---

## XQT 约定

### 开发阶段

当前 v0.x 开发期, 未到 v1.0, 在目标和范围已对齐后允许破坏性重构, 不需要兼容老接口: 改 API 可直接改, 改 recipe schema 可直接打破兼容, 删模块/改名/合并/拆分都可以. 取舍顺序: 清晰 > 简洁 > 方便 > 兼容.

### 核心契约

XQT 只关注模型本身: 压缩, 图变换, 导出适配, 误差分析, benchmark. XQT 不负责训练, QAT, finetune, distillation, KD/recovery, dataset/dataloader, training/evaluation provider. 需要梯度更新或任务验证的流程归 XDL 或第三方, 再把训练后的模型/checkpoint 或指标交给 XQT.

- 包内工程契约见 `xqt/FRAMEWORK.md`; 长期事实源见 `docs/md/XQT.md`.
- `xqt` 顶层导出及 XDL 适配入口 (`xdl_setup_to_xqt_context`, `xdl_checkpoint_to_xqt_context`, `load_checkpoint_into_model`) 进入 Provisional API; 子模块内部实现按 Internal 处理.
- 两种配置方式: **`XQTOptimizationSession`** (第一配置, Python 交互式 session) 和 **YAML workflow** (第二配置, `optimize_model("path.yaml")` 或 CLI). 不新增 CLI 参数解析库, JSON dict 硬编码 workflow 或其他配置路径.
- 不新增训练循环, trainer, training/evaluation provider, loss wrapper, 任务 registry 或 dataset/dataloader 构建.
- 新增 recipe 必须声明 `compression_axes` 和支持的硬件约束; 所有量化/剪枝/导出后必须能产出模型侧 report 或最小性能基准.
- 公共压缩/部署工具若可复用, 再考虑抽到 `tools/` 或 `xdl/`.
- 顶层示例脚本保持薄入口, 优先调 XQT 已有 helper; 可选依赖用 lazy import, 失败时抛清楚的 XQT 异常.
- 量化/部署/后端 adapter 对环境版本敏感, 先确认最新文档; TensorRT/OpenVINO/ncnn/MNN/ExecuTorch 以目标机器官方安装为准, XQT 只做 adapter 和 preflight; 真实 FP8 收益需支持硬件验证, synthetic smoke 仅本地闭环.

---

## 核心文档入口

以下文档是核心规范, 不是普通参考链接. `docs/md/README.md` 是 XDL MD 正文事实源. 改动对应范围前必须先读相应章节, 行为变化时同步更新.

- [docs/md/README.md](docs/md/README.md) - XDL MD 正文事实源, 推荐阅读顺序与各主题边界
- [docs/md/README.md#xdl-安装与验证](docs/md/README.md#xdl-安装与验证) - 安装, 环境验证, 可运行入口
- [docs/md/README.md#xdl-项目结构与使用说明](docs/md/README.md#xdl-项目结构与使用说明) - 框架定位, 核心分层, 训练入口, `CoreModel`/`Trainer` 生命周期
- [docs/md/README.md#xdl-config-系统说明](docs/md/README.md#xdl-config-系统说明) - YAML 配置系统, schema v1, `target + params`
- [docs/md/README.md#xdl-api-稳定边界](docs/md/README.md#xdl-api-稳定边界) - Stable/Provisional/Internal API 边界与兼容策略
- [docs/md/README.md#xdl-html-阅读页样式规范](docs/md/README.md#xdl-html-阅读页样式规范) - 自有 HTML 阅读页统一样式, 主题 token, 交互规范
- [docs/md/README.md#xdl-模块功能边界速查](docs/md/README.md#xdl-模块功能边界速查) - 各源码子模块职责速查
- [docs/md/README.md#xdl-当前优化方向](docs/md/README.md#xdl-当前优化方向) - 框架后续优化方向
- [xdl/USAGE.md](xdl/USAGE.md) - 随 wheel 分发的单文件用法摘要, `xdl-usage` 查看
- [xdl/trainer/README.md](xdl/trainer/README.md) - 训练器子模块, 手动优化, 梯度累积 helper, 日志命名, batch 迁移
- [docs/AGENTS.md](docs/AGENTS.md) - 仓库文档目录规范

---

## 子模块文档

### 顶层目录

- [config/AGENTS.md](config/AGENTS.md) - 训练与运行配置目录
- [data/AGENTS.md](data/AGENTS.md) - 本地数据目录 (数据集统一放 `data/`, 数据本体不入库)
- [docs/AGENTS.md](docs/AGENTS.md) - 仓库文档目录
- [downloads/AGENTS.md](downloads/AGENTS.md) - 下载产物目录
- [examples/AGENTS.md](examples/AGENTS.md) - 示例脚本目录
- [infer/AGENTS.md](infer/AGENTS.md) - 推理脚本目录
- [learn/AGENTS.md](learn/AGENTS.md) - 学习与实验目录
- [others/AGENTS.md](others/AGENTS.md) - 其他资产目录
- [research/AGENTS.md](research/AGENTS.md) - 研究资料目录
- [scripts/AGENTS.md](scripts/AGENTS.md) - 仓库维护脚本目录
- [tests/AGENTS.md](tests/AGENTS.md) - 测试目录
- [third_party/AGENTS.md](third_party/AGENTS.md) - 第三方代码与资产目录
- [tools/AGENTS.md](tools/AGENTS.md) - 数据与工程工具目录
- [train/AGENTS.md](train/AGENTS.md) - 训练入口目录 (预训练/后训练划分)
- [xdl/AGENTS.md](xdl/AGENTS.md) - XDL 框架源码根目录
- [xqt/AGENTS.md](xqt/AGENTS.md) - 量化与实验脚本目录

### XDL 核心子模块

- [xdl/callbacks/AGENTS.md](xdl/callbacks/AGENTS.md) - 回调系统
- [xdl/config/AGENTS.md](xdl/config/AGENTS.md) - 配置构建子模块
- [xdl/dataset/AGENTS.md](xdl/dataset/AGENTS.md) - 数据集子模块
- [xdl/loss/AGENTS.md](xdl/loss/AGENTS.md) - 损失函数子模块
- [xdl/metric/AGENTS.md](xdl/metric/AGENTS.md) - 评估指标子模块
- [xdl/model/AGENTS.md](xdl/model/AGENTS.md) - 模型架构子模块
- [xdl/post_training/AGENTS.md](xdl/post_training/AGENTS.md) - 后训练子模块
- [xdl/optimizer/AGENTS.md](xdl/optimizer/AGENTS.md) - 优化器子模块
- [xdl/scheduler/AGENTS.md](xdl/scheduler/AGENTS.md) - 学习率调度器子模块
- [xdl/trainer/AGENTS.md](xdl/trainer/AGENTS.md) - 训练器核心子模块
- [xdl/utils/AGENTS.md](xdl/utils/AGENTS.md) - 注册系统与工具子模块
