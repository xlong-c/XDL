# XDL — 模块化深度学习框架

PyTorch 深度学习框架，组件注册系统 + 回调生命周期。Python 3.8+, PyTorch 1.12+, CUDA 可选。

## 行为准则

- **语言**：所有推理和回答均使用中文
- 做非 trivial 改动前，先用 `grep` / `find` / Agent 搜索代码库，理解现有模式再动手
- 多文件重构或架构级变更时，先进入规划模式（EnterPlanMode）制定方案
- 不确定文件位置或调用关系时，优先搜索而非猜测
- 涉及第三方库用法、API 变更、最佳实践等外部知识时，先用 WebSearch 查最新文档

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

- **所有组件必须通过注册系统注册**：`@MODEL.register_module()` / `@DATASET.register_module()` 等
- 支持六种注册类型：MODEL, DATASET, OPTIMIZER, SCHEDULER, LOSS, METRIC
- **回调优先级**：数值越小越先执行，未指定时默认 999
- 所有函数必须加类型注解

## 三种使用方式

1. **纯代码**（VAE/GAN 示例）：手动实例化所有组件，`python train_VAE.py`
2. **YAML 配置（推荐）**：`setup = setup_from_yaml('config/xxx.yaml')` → 返回 `TrainSetup` dataclass，一行拿到 model/optimizer/train_loader 等全部组件
3. **DeepSpeed 分布式**：`deepspeed train_script.py --deepspeed config/deepspeed_config.json`

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
