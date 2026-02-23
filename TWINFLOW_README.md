# TwinFlow 训练器使用指南

## 概述

本项目提供了基于 **TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows** 论文的训练实现。

TwinFlow 是一种用于扩散模型加速的框架，通过自对抗流（Self-adversarial Flows）实现高质量的1步/少步生成。

- **论文**: https://arxiv.org/abs/2512.05150
- **项目页**: https://zhenglin-cheng.com/twinflow

## 核心特性

1. **自对抗流 (Self-adversarial Flows)**
   - 通过负时间分支创建内部双轨迹
   - 无需外部鉴别器或冻结教师模型

2. **速度场修正 (Velocity Field Rectification)**
   - 最小化真实轨迹和假轨迹之间的速度场差异
   - 逐步将模型转变为1步/少步生成器

3. **单模型简洁性**
   - 无需辅助网络
   - 可扩展到20B+参数模型

## 文件说明

### 1. `train_TwinFlow.py` - MNIST 训练示例

这是一个完整的 TwinFlow 训练示例，使用 MNIST 数据集。

**特点**:
- 简化的 Diffusion UNet 模型
- 完整的训练、验证、采样流程
- 支持条件生成（数字标签）
- 生成训练过程动画

**运行**:
```bash
python train_TwinFlow.py
```

**输出**:
- 检查点: `./others/checkpoints/twinflow/`
- 日志: `./others/logs/twinflow/`
- 动画: `./others/animations/twinflow/`
- 最终样本: `./others/checkpoints/twinflow/final_samples.png`

### 2. `train_TwinFlow_Diffusers.py` - Diffusers 模型训练

这是一个用于训练 Hugging Face Diffusers 模型的框架。

**支持模型**:
- Stable Diffusion 1.5/2.0/2.1
- Stable Diffusion XL
- 其他基于 Diffusers 的扩散模型

**特点**:
- 支持从 HuggingFace Hub 加载预训练模型
- 支持各种文本到图像数据集
- 支持混合精度训练
- 支持梯度累积

**使用示例**:

```bash
# 使用 Pokemon 数据集训练
python train_TwinFlow_Diffusers.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1 \
    --dataset_name lambdalabs/pokemon-blip-captions \
    --resolution 512 \
    --batch_size 4 \
    --max_epochs 100 \
    --learning_rate 2e-4 \
    --output_dir ./outputs/twinflow-pokemon

# 使用本地数据集
python train_TwinFlow_Diffusers.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --train_data_dir ./data/my-images \
    --resolution 512 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_epochs 50
```

**主要参数**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pretrained_model_name_or_path` | 预训练模型路径或 HuggingFace 模型ID | 必填 |
| `--dataset_name` | 数据集名称 | None |
| `--train_data_dir` | 本地训练数据目录 | None |
| `--resolution` | 图像分辨率 | 512 |
| `--batch_size` | 批次大小 | 4 |
| `--max_epochs` | 最大训练轮数 | 100 |
| `--learning_rate` | 学习率 | 2e-4 |
| `--ema_decay_rate` | EMA衰减率 | 0.99 |
| `--estimate_order` | RCGM估计阶数 | 2 |
| `--no_twinflow` | 禁用TwinFlow | False |

## 核心类说明

### `TwinFlowCoreModel` (CoreModel 子类)

核心模型类，整合扩散模型和 TwinFlow 训练逻辑。

**主要方法**:
- `training_step()`: 执行训练步骤，使用 TwinFlow 计算损失
- `validation_step()`: 执行验证，生成样本并计算指标
- `generate_samples()`: 从潜在空间生成样本
- `configure_optimizers()`: 配置优化器

### `TwinFlowTrainer` (from `xdl.model.generate.twinflow`)

TwinFlow 训练器，实现论文中的核心算法。

**主要功能**:
- `training_step()`: 计算 TwinFlow 损失（包括 L_base, L_adv, L_rectify）
- `sampling_loop()`: 使用 UCGM 采样器进行生成
- `prepare_inputs()`: 准备训练输入（添加噪声、构造目标）
- `update_ema()`: 更新 EMA 模型

**关键算法**:
1. **自对抗流 (Self-adversarial Flows)**:
   - 使用负时间分支 `t ∈ [-1, 0]` 创建假轨迹
   - 通过 `fake_s, fake_v = forward(x_t, -t, -t)` 计算假样本

2. **速度场修正 (Velocity Rectification)**:
   - 计算真实速度 `real_v` 和假速度 `fake_v`
   - 最小化速度场差异 `Δv = fake_v - real_v`
   - 更新目标 `target = target + ratio * (pred_w_c - pred_wo_c)`

3. **分布匹配 (Distribution Matching)**:
   - 对齐生成流的梯度方向
   - 计算 `L_rectify = loss(F_fake, (F_fake - F_grad).detach())`

## 训练流程

### 1. 数据准备

```python
# 加载数据
train_loader, val_loader = get_mnist_dataloaders(batch_size=128)

# 或使用自定义数据集
dataset = load_dataset("lambdalabs/pokemon-blip-captions")
```

### 2. 模型创建

```python
# 创建 TwinFlow 核心模型
model = TwinFlowCoreModel(
    data_dim=784,
    hidden_dim=128,
    time_embed_dim=64,
    num_classes=10,
    ema_decay_rate=0.99,
    estimate_order=2,
    using_twinflow=True,
    learning_rate=2e-4,
)
```

### 3. 训练器配置

```python
# 创建 xdl Trainer
trainer = Trainer(
    max_epochs=20,
    device='cuda',
    precision='bf16',
    callbacks=[
        LoggingCallback(log_frequency=50),
        TensorBoardCallback(),
    ],
)

# 配置日志
trainer.setup_logger(
    experiment_name='twinflow_mnist',
    log_dir='./logs',
    checkpoint_dir='./checkpoints',
    monitor='val_recon_loss',
    mode='min',
    save_top_k=3,
)
```

### 4. 开始训练

```python
# 开始训练
trainer.fit(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    val_check_interval=500,
)
```

### 5. 生成样本

```python
# 生成样本
model.eval()
samples = model.generate_samples(
    n_samples=16,
    labels=torch.randint(0, 10, (16,)),
)

# 保存图像
save_image(samples, 'generated_samples.png')
```

## 训练技巧

### 1. 学习率调度

```python
# 使用 Cosine Annealing 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epochs, eta_min=1e-6
)
```

### 2. 梯度裁剪

```python
# 在 training_step 中使用梯度裁剪
self.clip_gradients(model=self.model, gradient_clip_val=1.0)
```

### 3. EMA 更新

```python
# TwinFlow 内置 EMA 更新
self.twinflow_trainer.update_ema(self.model)
```

### 4. 内存优化

```python
# 使用梯度检查点
self.unet.enable_gradient_checkpointing()

# 使用混合精度训练
trainer = Trainer(precision='bf16')
```

## 常见问题

### 1. OOM (显存不足)

**解决方案**:
- 减小批次大小 (`--batch_size`)
- 启用梯度累积 (`--gradient_accumulation_steps`)
- 减小分辨率 (`--resolution`)
- 使用梯度检查点

### 2. 训练不稳定

**解决方案**:
- 减小学习率
- 增大EMA衰减率 (`--ema_decay_rate=0.995`)
- 使用梯度裁剪

### 3. 生成质量差

**解决方案**:
- 增加训练轮数
- 增大模型容量
- 调整 `--enhanced_ratio` 参数
- 使用多步采样 (`num_inference_steps > 1`)

## 引用

如果您在研究中使用了本代码，请引用 TwinFlow 论文：

```bibtex
@article{cheng2025twinflow,
  title={TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows},
  author={Cheng, Zhenglin and Sun, Peng and Li, Jianguo and Lin, Tao},
  journal={arXiv preprint arXiv:2512.05150},
  year={2025}
}
```

## 许可证

本项目遵循原始论文和代码的许可证。

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
