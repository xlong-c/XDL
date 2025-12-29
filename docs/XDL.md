# XDL 项目文件大纲

本文档整理了 xdl 项目下所有文件的功能说明,帮助快速理解项目结构。

---

## 核心模块 (xdl/)

### 主模块
- **__init__.py** - xdl 框架初始化模块,导出各个子模块
- **README.md** - 项目说明文档

---

## 回调模块 (xdl/callbacks/)

回调系统提供了完整的训练生命周期钩子管理,支持在训练的各个阶段插入自定义逻辑。

### 核心功能

#### 1. 生命周期钩子
- **训练阶段** - `on_train_start`, `on_train_end`, `on_train_epoch_start/end`, `on_train_batch_start/end`
- **验证阶段** - `on_validation_start/end`, `on_validation_epoch_start/end`, `on_validation_batch_start/end`
- **测试阶段** - `on_test_start/end`, `on_test_epoch_start/end`, `on_test_batch_start/end`
- **预测阶段** - `on_predict_start/end`, `on_predict_epoch_start/end`, `on_predict_batch_start/end`
- **检查点** - `on_save_checkpoint`, `on_load_checkpoint`
- **异常处理** - `on_exception`

#### 2. 回调管理特性
- **优先级控制** - 支持按优先级排序回调执行顺序
- **错误隔离** - 单个回调失败不影响其他回调执行
- **状态管理** - 支持回调状态保存和恢复
- **执行统计** - 记录每个回调的执行时间和错误信息

#### 3. 内置回调
- **日志记录** - Console, Loguru, TensorBoard, WandB
- **监控** - 设备统计,学习率监控,模型摘要,训练计时
- **训练控制** - 模型检查点,早停机制
- **进度显示** - Tqdm 进度条,支持总进度和分阶段进度

### 核心基类
- **base.py** - Callback 基类,所有自定义回调的父类,定义完整的生命周期钩子
- **callback_list.py** - 回调管理器,统一管理所有训练回调,支持错误隔离和执行统计

### 日志回调
- **console_callback.py** - 控制台日志回调,将训练和验证指标输出到控制台
- **logging_callback.py** - loguru 日志回调,记录训练指标到日志文件,支持滚动和压缩
- **tensorboard_callback.py** - TensorBoard 日志回调,记录指标到 TensorBoard 可视化
- **wandb_callback.py** - WandB 日志回调,记录指标到 Weights & Biases

### 监控回调
- **device_stats_monitor.py** - 设备统计监控,监控 CPU/GPU/内存使用情况
- **learning_rate_monitor.py** - 学习率监控,记录优化器学习率变化
- **layer_monitor.py** - 网络层监控,监控特定层的权重和梯度分布
- **model_summary.py** - 模型结构摘要,显示模型层数和参数统计
- **timer.py** - 训练计时回调,记录各阶段时间消耗和 ETA

### 训练控制回调
- **model_checkpoint.py** - 模型检查点,自动保存最佳模型和最后状态
- **early_stopping.py** - 早停机制,监控指标并在满足条件时停止训练

### 其他回调
- **tqdm_callback.py** - Tqdm 进度条,显示训练进度,支持多阶段管理
- **lambda_callback.py** - Lambda 回调,提供轻量级的自定义回调功能

### 模块文件
- **__init__.py** - 回调模块导出文件
- **README.md** - 回调模块使用说明

---

## 损失函数模块 (xdl/loss/)

提供各种深度学习损失函数实现。

### 核心文件
- **__init__.py** - 损失函数模块初始化和统一注册

### 损失函数实现
- **bce.py** - 二元交叉熵损失 (BCELoss, BCEWithLogitsLoss)
- **cross_entropy.py** - 交叉熵损失
- **focal_loss.py** - Focal Loss,用于处理类别不平衡
- **mae.py** - 平均绝对误差损失 (L1Loss, MAELoss)
- **mse.py** - 均方误差损失 (MSELoss)
- **smooth_l1.py** - Smooth L1 损失 (HuberLoss)

---

## 评估指标模块 (xdl/metric/)

- **__init__.py** - 评估指标模块初始化
- **metrics.py** - 评估指标实现

---

## 模型模块 (xdl/model/)

### 卷积模型
- **resnet.py** - ResNet 系列网络 (ResNet-18/34/50/101/152)
- **vgg.py** - VGG 系列网络
- **vit.py** - Vision Transformer 系列 (ViT-Tiny/Small/Base/Large/Huge)

### 分割模型
- **segment/fatt.py** - FATT 分割模型

### 模块文件
- **__init__.py** - 模型模块初始化

---

## 优化器模块 (xdl/optimizer/)

- **__init__.py** - 优化器模块初始化
- **adam.py** - Adam 优化器
- **adamw.py** - AdamW 优化器
- **muon.py** - Muon 优化器
- **rmsprop.py** - RMSprop 优化器
- **sgd.py** - SGD 随机梯度下降

---

## 学习率调度器模块 (xdl/scheduler/)

- **__init__.py** - 调度器模块初始化
- **cosine_annealing_lr.py** - 余弦退火学习率调度
- **cosine_annealing_warm_restarts.py** - 余弦退火重启调度
- **exponential_lr.py** - 指数衰减学习率调度
- **multi_step_lr.py** - 多步学习率调度
- **step_lr.py** - 阶梯学习率调度

---

## 训练器模块 (xdl/trainer/)

### 核心训练组件
- **coreModel.py** - 核心模型基类,用户继承并实现训练和验证逻辑
- **trainer.py** - 训练器主类,管理训练循环、设备和优化器
- **trainer_state.py** - 训练状态管理,统一管理训练过程中的状态信息

---

### CoreModel 核心功能

CoreModel 是用户模型的基类,提供完整的训练接口和生命周期管理。

#### 1. 必须实现的方法
- **training_step(batch, batch_idx)** - 单步训练逻辑
  - 手动管理前向传播、反向传播和优化步骤
  - 通过 `self.log()` 记录训练指标
  - 支持 `self.manual_backward()` 和 `self.clip_gradients()`

- **validation_step(batch, batch_idx)** - 单步验证逻辑
  - 计算验证指标
  - 通过 `self.log()` 记录验证指标

- **configure_optimizers()** - 配置优化器和调度器
  - 支持单个优化器、多个优化器
  - 支持优化器+调度器组合
  - 返回格式灵活: `optimizer`, `[opt1, opt2]`, `[optimizers], [schedulers]`

#### 2. 指标管理
- **self.log(name, value)** - 记录单个指标
- **self.log_metrics(dict)** - 批量记录指标
- **current_metrics** - 获取当前指标
- **epoch_avg** - 获取当前 epoch 平均指标
- **last_epoch_avg** - 获取上一个 epoch 平均指标

#### 3. 训练状态
- **current_epoch** - 当前 epoch 数
- **total_train_steps** - 总训练步数
- **train_steps_epoch** - 当前 epoch 的训练步数
- **is_main_process()** - 判断是否为主进程(分布式训练)

#### 4. 设备和梯度
- **device** - 自动获取模型所在设备
- **manual_backward(loss)** - 兼容 Accelerate 的反向传播
- **clip_gradients()** - 梯度裁剪,支持 norm 和 value 两种算法

#### 5. 检查点管理
- **save_checkpoint()** - 保存模型状态
- **load_checkpoint()** - 加载模型状态
- **load_from_checkpoint()** - 类方法,直接从文件实例化模型
- 支持多种格式: pt, pth, st, safetensors, accelerator

#### 6. 生命周期钩子
用户可重写的钩子方法:
- 训练阶段: `on_train_start/end`, `on_train_epoch_start/end`, `on_train_step_start/end`
- 验证阶段: `on_validation_start/end`, `on_validation_epoch_start/end`
- 测试阶段: `on_test_start/end`, `on_test_epoch_start/end`
- 预测阶段: `on_predict_start/end`, `on_predict_epoch_start/end`
- 检查点: `on_save_checkpoint`, `on_load_checkpoint`
- 设备变更: `on_device_change`

#### 7. 推理方法
- **inference(data)** - 推理/采样方法,专门用于生成式模型
- **predict_step(batch)** - 单步预测逻辑

---

### Trainer 核心功能

Trainer 是训练引擎,管理完整的训练流程和资源。

#### 1. 初始化参数
- **max_epochs** - 最大训练轮数
- **device** - 设备选择 ('cpu', 'cuda', 'cuda:0' 等)
- **precision** - 混合精度 ('16', 'bf16', '32')
- **gradient_accumulation_steps** - 梯度累积步数
- **grad_clip_max_norm** - 梯度裁剪最大范数
- **callbacks** - 回调函数列表
- **accelerate_config** - Accelerate 分布式训练配置

#### 2. 核心方法
- **fit(model, train_dataloader, val_dataloader)** - 执行完整训练
  - 支持 Accelerate 分布式训练
  - 自动处理设备分配
  - 支持虚拟 epoch 模式(灵活的验证间隔)
  - 支持 inference 推理采样

- **test(model, test_dataloader)** - 执行测试
- **setup_logger()** - 快速配置常用回调
  - TensorBoard 日志
  - 模型检查点
  - Tqdm 进度条
  - 控制台日志

#### 3. 训练流程管理
- **训练循环** - 自动管理 epoch 和 step 循环
- **验证调度** - 支持按 epoch 或按 step 验证
- **梯度累积** - 自动处理梯度累积逻辑
- **混合精度** - 支持 fp16/bf16 训练
- **分布式训练** - 集成 Accelerate,支持多 GPU/多节点

#### 4. 状态访问
- **global_step** - 全局训练步数
- **current_epoch** - 当前 epoch
- **should_stop** - 是否应该停止训练(早停标志)
- **steps_per_epoch** - 每个 epoch 的步数(支持虚拟 epoch)
- **device** - 当前使用的设备
- **accelerator** - Accelerate 实例
- **callback_metrics** - 回调指标字典

#### 5. 回调系统集成
- **callback_list** - 回调列表管理器
- **callbacks** - 回调列表(向后兼容)
- 自动在训练各阶段调用相应的回调钩子
- 支持回调优先级排序

#### 6. 设备管理
- **is_main_process()** - 判断是否为主进程
- 自动检测和使用可用的 GPU
- 支持 Accelerate 分布式设备配置
- 自动将数据和模型移动到正确设备

---

### TrainerState 核心功能

训练状态管理类,统一管理训练过程中的所有状态信息。

#### 1. 基础状态
- **global_step** - 全局训练步数
- **current_epoch** - 当前 epoch
- **max_epochs** - 最大 epoch 数
- **should_stop** - 停止训练标志

#### 2. 指标存储
- **callback_metrics** - 回调指标
- **logged_metrics** - 已记录指标
- **progress_bar_metrics** - 进度条指标

#### 3. 训练历史
- **training_history** - 每个 epoch 的训练结果
- **validation_history** - 验证历史记录

#### 4. 状态操作
- **epoch_start(epoch)** - epoch 开始,重置指标
- **epoch_end(epoch, metrics)** - epoch 结束,保存历史
- **add_metric(name, value, where)** - 添加指标到指定位置
- **get_metric(name, where)** - 获取指标值
- **reset()** - 重置所有状态
- **to_dict()** / **from_dict()** - 状态序列化

### 文档文件
- **README.md** - 训练器模块使用说明
- **hu_accelerate.md** - Accelerate 使用指南

---

## 工具模块 (xdl/utils/)

- **__init__.py** - 工具模块初始化
- **checkpoint.py** - 检查点工具,提供格式无关的保存和加载
- **registry.py** - 注册表系统,实现名称到对象的映射管理
- **tools.py** - 通用工具函数集合
- **weight.py** - 权重处理工具

---

## 项目整体架构

### 核心设计理念
1. **模块化设计** - 各功能模块独立,通过注册表系统连接
2. **回调驱动** - 训练过程通过回调系统扩展
3. **配置灵活** - 支持多种配置方式和后端
4. **易于扩展** - 基于继承和装饰器的扩展机制

### 训练流程
```
用户模型 (CoreModel)
    ↓
配置优化器和调度器
    ↓
创建训练器 (Trainer)
    ↓
添加回调 (Callbacks)
    ↓
执行训练 (fit)
```

### 关键特性
- 支持 Accelerate 分布式训练
- 完整的检查点管理
- 丰富的日志和监控功能
- 灵活的模型注册系统
- 全面的钩子机制

---

## 快速开始指南

### 1. 定义模型
继承 `CoreModel` 并实现必要方法:
- `training_step()` - 单步训练逻辑
- `validation_step()` - 单步验证逻辑
- `configure_optimizers()` - 配置优化器

### 2. 创建训练器
```python
trainer = Trainer(
    max_epochs=10,
    device='cuda',
    callbacks=[...]
)
```

### 3. 执行训练
```python
trainer.fit(model, train_dataloader, val_dataloader)
```

---

## 使用技巧

### 1. 查看注册的组件
```python
from xdl.utils import inspect_model, inspect_optimizer
inspect_model('resnet18')
inspect_optimizer('Adam')
```

### 2. 使用 Registry 创建组件
```python
from xdl.utils import build_model
model = build_model('resnet18')(num_classes=10)
```

### 3. 配置日志回调
```python
trainer.setup_logger(
    enable_tensorboard=True,
    enable_checkpoint=True,
    enable_tqdm=True
)
```

---

## 注意事项

- 所有代码使用 UTF-8 编码
- 注释使用中文,专业术语保持英文
- 遵循现有代码风格
- 非代码文件(日志、权重等)存放于 `others/` 目录

---

最后更新: 2025-12-28
