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
  analysis/    中间表征解析与可解释性分析
  callbacks/   训练生命周期扩展
  config/      YAML 到 TrainSetup 的构建链
  dataset/     数据集, transform, collate
  loss/        预训练通用损失函数
  post_training/ 后训练组件, 按 losses/callbacks/checkpoint 分层
  metric/      评估指标
  model/       模型与工厂函数
  optimizer/   优化器
  scheduler/   学习率调度器
  trainer/     CoreModel / Trainer / TrainSetupModel
  task/        预训练和后训练算法任务
  utils/       registry, checkpoint, 通用工具
```

## `xdl/analysis`

职责:

- 提供离线分析和研究脚本可复用的模型中间表征解析工具
- 承载 activation capture, linear probe, concept probe / TCAV, Grad-CAM, attention rollout 这类不接管训练主逻辑的分析能力

不负责:

- 训练主循环
- 数据集构建
- 作为 callback 直接接管训练期逻辑

关键点:

- 这里优先放纯 PyTorch,轻依赖的分析 API
- 训练期如需抓特征, 应通过 callback 或 forward hook 采样后再调用这里的工具
- 这层更接近研究和诊断基座, 不是稳定冻结的训练主入口
- callback 侧如果需要在验证期自动落分析产物, 例如 feature capture 或 attention rollout, 只负责时机和落盘; 具体算法仍应复用 `xdl.analysis`
- 结果导出也保持轻量, 只提供 JSON / CSV / Markdown 写出 helpers, 不引入新的 report 框架

## `xdl/callbacks`

职责:

- 日志, 进度条, 检查点, 早停, 监控等横切逻辑
- 跟随训练生命周期执行
- 训练期或验证期的轻量观测与产物落盘,例如 preview, feature capture

不负责:

- 具体模型训练逻辑
- 优化器步进
- 数据加载

关键点:

- 优先级数值越小越先执行
- callback 更适合作为观察者, 不应承载主训练逻辑
- 中间特征抓取这类能力如果需要训练期时机, 放 callback; 真正的分析算法留在 `xdl.analysis`

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

- 提供预训练通用的单个或组合损失函数
- 通过 registry 接入配置系统

关键点:

- 新增 loss 后要在 `__init__.py` 中集中注册
- 多 loss 配置会由 `build_loss()` 构建成 `WeightedLoss`
- 蒸馏/偏好优化等后训练 loss 不在这里, 见 `xdl/post_training`

## `xdl/post_training`

职责:

- 收拢基于 pretrain checkpoint 的后训练组件,并按依赖层次分组:
  - `losses/`: 偏好优化, 通用蒸馏和扩散 few-step 蒸馏的纯目标函数
  - `callbacks/`: rollout, 参考模型和 adapter 状态保存等训练期扩展
  - `checkpoint/`: checkpoint 合并等不依赖 Trainer 的产物处理
- 根包只提供懒加载公开入口, 不在 import 阶段加载 callback 栈

关键点:

- 旧的 `xdl.post_training.preference_loss` 等模块路径保持兼容.
- LOSS registry 只导入 `xdl.post_training._registry`, 不触发 callbacks.
- 从零训练不经过本子模块; 端到端参考入口为 `train/posttrain/train_GRPO.py`

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

## `xdl/task`

职责:

- 承载具体训练算法任务对 `CoreModel` 的实现, 按训练阶段分为
  `pretrain/` 和 `posttrain/`.
- `xdl/task/pretrain/tbsm.py` 提供 TBSM 训练任务, 复用
  `xdl/model/generate/tbsm.py` 中的模型侧组件.

不负责:

- 通用训练循环, 由 `xdl.trainer` 负责.
- 通用模型组件, 由 `xdl.model` 负责.

关键点:

- 算法任务可以继承 `CoreModel`, 但不应把算法特定逻辑加入 `Trainer` 或
  `CoreModel`.
- `xdl.trainer.TBSMCoreModel` 作为兼容导出保留, 新代码优先从
  `xdl.task.pretrain` 导入.
## `xdl/trainer`

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
