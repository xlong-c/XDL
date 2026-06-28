# XDL 模块边界

本文承接 `XDL` 子模块职责的架构层正文. 它定义各源码子模块负责什么, 不负责什么, 以及依赖方向应该怎样保持清楚.

## 负责什么

- 说明 `xdl/` 各核心子模块的职责边界.
- 说明模块之间的推荐依赖方向.
- 说明哪些变化属于高影响结构改动.

## 不负责什么

- 不重复完整训练工作流.
- 不展开每个模块的全部 API 细节.
- 不替代包内子目录的局部 `AGENTS.md`.

## 总览

```text
xdl/
  callbacks/   训练生命周期扩展
  config/      YAML 到 TrainSetup 的构建链
  dataset/     数据集, transform, collate
  loss/        损失函数
  metric/      评估指标
  model/       模型与工厂函数
  optimizer/   优化器
  scheduler/   学习率调度器
  trainer/     CoreModel / Trainer / TrainSetupModel
  utils/       registry, checkpoint, 通用工具
```

## `xdl/callbacks`

职责:

- 日志, 进度条, 检查点, 早停, 监控等横切逻辑
- 跟随训练生命周期执行

不负责:

- 具体模型训练逻辑
- 优化器步进
- 数据加载

关键点:

- 优先级数值越小越先执行
- callback 更适合作为观察者, 不应承载主训练逻辑

## `xdl/config`

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

## `xdl/dataset`

职责:

- 数据集定义
- transform / collate 相关能力
- 通过 registry 对外暴露

不负责:

- 训练循环
- 模型前向

关键点:

- 常见容器 batch 会由 Trainer 递归迁移; 第三方自定义对象需要自行处理设备
- 可选依赖应优雅降级

## `xdl/loss`

职责:

- 提供单个或组合损失函数
- 通过 registry 接入配置系统

关键点:

- 新增 loss 后要在 `__init__.py` 中集中注册
- 多 loss 配置会由 `build_loss()` 构建成 `WeightedLoss`

## `xdl/metric`

职责:

- 提供训练与验证指标
- 按配置构建指标列表

关键点:

- 指标通常需要稳定的 `update / compute` 或可调用语义
- 注册名与导出要同步维护

## `xdl/model`

职责:

- 提供模型类与工厂函数
- 通过 registry 暴露给纯代码路径和 YAML 路径

不负责:

- 训练循环编排
- 日志和检查点

关键点:

- 当前源码中包含分类, ViT, 生成, 分割和底层超分模型
- 新模型应在 `__init__.py` 中集中注册

## `xdl/optimizer`

职责:

- 自定义优化器封装
- 与 PyTorch optimizer 一起统一进入构建链

关键点:

- `build_optimizer()` 支持 `target_modules` 和 `param_groups`
- 需要明确参数选择范围和状态初始化语义

## `xdl/scheduler`

职责:

- 学习率调度器与工厂函数
- 为训练流程提供统一调度入口

关键点:

- 要区分 step 级还是 epoch 级调用语义
- 配置侧由 `build_scheduler()` 负责实例化

## `xdl/trainer`

职责:

- `CoreModel`: 任务逻辑抽象
- `Trainer`: 训练循环编排
- `TrainSetupModel`: 把配置流构建的外部组件桥接到 `CoreModel`

关键点:

- 稳定公共入口是 `from xdl.trainer import CoreModel, Trainer, TrainSetupModel`
- `Trainer.fit()` 先调用 `model.setup("fit")`
- `CoreModel.training_step()` 是手动优化模式
- 手动累积优先用 `micro_step` / `is_accumulation_boundary` 等公开 helper
- `self.log("loss", value, prefix="train")` 会生成 `train_loss`
- callback 在这里被统一调度

## `xdl/utils`

职责:

- registry
- checkpoint
- tiling
- 权重与通用辅助函数

关键点:

- 这是基础设施层, 改动影响面大
- registry 只做名字到对象的映射, 不掺配置解析
- 自定义组件稳定接入方式是 `register_*("Name")(ClassOrFunction)`

## 依赖方向

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

## 什么改动算高影响

下列变化不属于普通文案修订, 应同时更新相关架构文档和知识图谱:

- 公开导出符号重命名
- 子模块职责迁移或目录重组
- `Trainer` / `CoreModel` 生命周期变化
- config 构建链新增平行入口
- registry 接入方式变化

相关页面:

- [xdl.md](xdl.md)
- [api-boundary.md](api-boundary.md)
- [../README.md#xdl-模块功能边界速查](../README.md#xdl-模块功能边界速查)
