# XDL — 模块化深度学习框架

PyTorch 深度学习框架，组件注册系统 + 回调生命周期。Python 3.8+, PyTorch 1.12+, CUDA 可选。

## 行为准则

- **语言**：所有推理和回答均使用中文
- 做非 trivial 改动前，先用 `grep` / `find` / Agent 搜索代码库，理解现有模式再动手
- 多文件重构或架构级变更时，先进入规划模式（EnterPlanMode）制定方案
- 不确定文件位置或调用关系时，优先搜索而非猜测
- 涉及第三方库用法、API 变更、最佳实践等外部知识时，先用 WebSearch 查最新文档
- **脚本参数**：Python 脚本不要使用命令行参数解析库（如 `argparse`/`args`），优先使用 YAML 配置或代码内显式配置

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

## 核心约束

- **所有组件必须通过注册系统注册**：在对应 `__init__.py` 中用 `register_*("Name")(Class)` 集中注册（非装饰器模式）
- 支持六种注册类型：MODEL, DATASET, OPTIMIZER, SCHEDULER, LOSS, METRIC
- **回调优先级**：数值越小越先执行，未指定时默认 999
- 所有函数必须加类型注解

## 三种使用方式

1. **纯代码**（VAE/GAN 示例）：手动实例化所有组件，`python train_VAE.py`
2. **YAML 配置（推荐）**：`setup = setup_from_yaml('config/xxx.yaml')` → 返回 `TrainSetup` dataclass，一行拿到 model/optimizer/train_loader 等全部组件
3. **DeepSpeed 分布式**：`deepspeed train_script.py --deepspeed config/deepspeed_config.json`

## XDL 训练脚本接入备忘

编写新的训练入口时，优先先看 `train_VAE.py`、`train_TwinFlow.py`、`examples/*finetune.py` 和 `xdl/trainer/{trainer.py,coreModel.py}`。这几个文件能最快说明 XDL 的真实生命周期。

- `Trainer.fit()` 会先调用 `model.setup("fit")`，再执行 Trainer 的设备/优化器 setup；大模型、diffusers pipeline、PEFT LoRA 等重组件适合在 `CoreModel.setup()` 中懒加载。
- `CoreModel.training_step()` 是**手动优化模式**，Trainer 不会自动 `zero_grad/backward/step`。训练步里需要自行调用 `optimizer.zero_grad()`、`self.manual_backward(loss)`、`self.clip_gradients(...)`、`optimizer.step()`，并用 `self.log()` 记录指标。
- `configure_optimizers()` 在 `setup()` 之后由 Trainer 调用；如果优化器依赖懒加载出来的模块，必须确保这些模块已经在 `setup()` 中初始化。
- 标准非 Accelerate 路径下，Trainer 只会把 `CoreModel.__dict__` 中的 `nn.Module` 属性迁移到设备；`diffusers.Pipeline` 不是 `nn.Module`，需要把底层 `vae/text_encoder/transformer` 注册成模块属性，或在模型钩子中显式 `pipe.to(device)`。
- Trainer 对 batch 的自动设备迁移只处理顶层 iterable 里的 tensor；字典、嵌套 list/dict、第三方 dataset 返回的复杂结构，建议在 `training_step()` 内显式 `.to(self.device)`。
- `on_train_step_start()` 会在 `training_step()` 前递增 `_total_train_steps`；自己实现梯度累积时要注意这个计数已经是当前 step。Trainer 构造参数里的 `gradient_accumulation_steps` 不会替代手动优化逻辑。
- 自定义保存优先用 Callback。XDL 的通用 `ModelCheckpoint` 会保存 `CoreModel` 的 state_dict/optimizer 状态；diffusers/PEFT LoRA 这类权重通常要写专门的 callback 调 `save_pretrained()` 或 `StableDiffusion3Pipeline.save_lora_weights()`。
- `inference_data` 会走验证/推理周期并调用 `CoreModel.inference(data)`，适合生成式模型的采样预览；没有 val loader 时也可以只传 prompts 做周期性采样。
- 对外部 `third_party/` 代码不要只信 README 路径，先用 `grep`/`find` 查真实文件；如果第三方目录不是 Python package，可在脚本里用受控的 `sys.path.insert()` 或 `importlib.util.spec_from_file_location()` 加载。
- 训练脚本参数继续遵守本仓库约束：不要引入 `argparse`，优先 YAML 配置；需要切换配置时可用环境变量指向 YAML。

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

- [xdl/callbacks/AGENTS.md](xdl/callbacks/AGENTS.md) — 回调系统详解
- [xdl/model/AGENTS.md](xdl/model/AGENTS.md) — 模型架构详解
