# XDL 当前优化方向

本文档只保留当前仍然有效的改进项，不重复过时问题，也不重新解释模块结构。

## P0 — 低风险立即可做

### 1. `xdl/model/AGENTS.md` 与当前代码继续对齐

原因：

- `xdl/model/` 已包含分类、ViT、生成、分割、低层超分等多类条目
- 模块文档应该持续跟上注册表真实状态

建议：

- 保持已注册条目清单准确
- 简化示例，避免文档再次膨胀成架构分析

### 2. 训练入口文档补最短主路径示例

原因：

- 当前最真实的接入方式其实是两条：纯代码路径和 `setup_from_yaml()` 路径
- 文档已说明方向，但还可以继续压缩成更短的复制即用示例

建议：

- 在 `README.md` 或 `docs/XDL.md` 中只保留一段最短 runnable 代码

## P1 — 质量提升

### 3. `xdl/dataset/hair/` 的毛发数据集增强逻辑去重

原因：

- `xdl/dataset/hair/` 下的毛发数据集仍可能存在共享增强逻辑

建议：

- 提取共享 transform 工厂
- 保持数据集文件关注自己的索引与标注逻辑

### 4. tests 覆盖继续向 trainer / callbacks / optimizer 扩展

原因：

- 当前配置相关链路相对清楚
- 训练编排、回调组合和优化器行为更值得补回归保护

建议：

- `trainer` 主路径
- callback 组合行为
- 自定义 optimizer 最小 step 行为

### 5. 文档与目录局部说明继续标准化

原因：

- 这轮已经给大部分目录补了 `AGENTS.md` / `CLAUDE.md`
- 还可以再统一章节顺序和粒度

建议：

- `职责`
- `当前内容`
- `修改约束`
- `验证建议`

## P2 — 中期增强

### 6. 配置系统继续补强 dataset / collate / task 级抽象

原因：

- 当前 config 系统已经能稳定构建组件
- 但 task 级差异仍主要留在训练脚本或 `CoreModel` 子类里

建议：

- 只在确有复用价值时上提抽象
- 避免把任务私有逻辑硬塞进 schema

### 7. `xdl/dataset/` 按数据形态继续补模板

原因：

- 当前 manifest 主路径已经基本成形
- `ImageFolderDataset` 和 `ImageTextSidecarDataset` 已覆盖纯图片目录与 `image + txt` 基础入口
- `ImageMaskSidecarDataset` 已覆盖常见 `images/000.png` 对应 `masks/000.png` 的分割入口
- 实际项目里经常先面对“数据怎么摆”，再决定任务类型

建议把 dataset 继续按两层补齐：

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
- 标准格式标注（COCO / YOLO / VOC / keypoint）

建议优先级：

1. 更通用的 `BasenameAlignedDataset`
   - 复用当前 sidecar helper
   - 覆盖 `image + json` / `image + label` 这类 sidecar 结构

2. 标准格式适配
   - `COCODetectionDataset`
   - `COCOSegmentationDataset`
   - keypoint 模板

落地原则：

- manifest 继续作为长期推荐主路径
- sidecar / image-folder 模板作为低门槛接入路径
- 不把所有数据组织方式都硬塞成一个超大 dataset 类
- 优先沉淀成可注册、可 YAML 构建、可测试的通用模板，而不是只在训练脚本里临时实现

### 8. 分布式与大模型接入路径继续沉淀

原因：

- 当前 Trainer 已支持 accelerate 方向
- 但更复杂的大模型、外部 pipeline、LoRA 保存路径仍偏工程化

建议：

- 优先在真实训练入口中沉淀稳定模式
- 成熟后再回抽到 callback 或 config 层
