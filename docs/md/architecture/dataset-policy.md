# XDL Dataset 模板规范

本文承接 `XDL` dataset 模板规划的架构层正文. 它定义 dataset 应该怎样分类, 优先复用哪些模板, 以及什么时候才值得新增新的 dataset 类.

## 负责什么

- 定义 dataset 模板选择原则.
- 定义样本语义形态和磁盘组织形态的主分类.
- 定义新增 dataset 模板时的落地规则.

## 不负责什么

- 不替代 HTML 结构图或概念说明页.
- 不重复完整训练 workflow.
- 不把所有 YAML 细节堆成操作手册.

## 数据存放约定

- 所有数据集和样本数据统一放仓库根目录的 `data/`
- 不新建顶层 `datasets/`, `raw/`, `resources/` 等平行目录
- `data/` 被 `.gitignore` 整体忽略, 数据本体不入库
- 配置引用本地数据时, 优先使用 `${xdl.abspath:${xdl.config_dir},data/...}`

## 规划原则

设计 dataset 时先分清两个维度:

- 样本语义形态: 单条样本返回什么
- 磁盘组织形态: 数据在磁盘上怎样摆放

推荐顺序:

1. 能写 manifest 时, 优先使用 `Record*` 模板
2. 数据已经是目录分类或纯图片目录时, 使用目录模板
3. 数据是同名文件对齐时, 使用 sidecar 模板
4. 只有字段解析或采样逻辑确实特殊时, 才新增薄 dataset 类
5. 新增模板必须集中注册并补基本行为测试

命名约定:

- 新代码优先使用 `Record*` / `ImageEdit*`
- 历史 `Manifest*` 类名和 registry 键继续保留为兼容别名

## 样本语义形态

| 语义形态 | 典型任务 | 推荐模板 |
| --- | --- | --- |
| `image` | 推理, 自监督, 特征提取 | `ImageFolderDataset` |
| `image + label` | 单标签分类 | `ImageFolderClassificationDataset`, `RecordClassificationDataset` |
| `image + target` | 回归, 质量估计 | `RecordRegressionDataset` |
| `image + labels` | 多标签分类 | `RecordMultiLabelClassificationDataset` |
| `image + mask` | 分割, depth | `RecordSegmentationDataset`, `ImageMaskSidecarDataset` |
| `image + boxes + labels` | 检测 | `RecordDetectionDataset` |
| `image + text` | caption, diffusion 微调 | `RecordImageTextDataset`, `ImageTextSidecarDataset` |
| `text (+ target_text)` | 文本分类, SFT | `RecordTextDataset` |
| `image_a + image_b (+ label)` | pair matching, 对比学习 | `RecordPairDataset` |
| `anchor + positive + negative` | metric learning | `RecordTripletDataset` |
| `source_image + target_image (+ reference_image + edit_mask + prompt)` | 图像编辑 | `ImageEditDataset` |
| 任意 dict record | 表格, 推荐, 时序窗口 | `RecordDataset` 或继承 `RecordDatasetBase` |

## 磁盘组织形态

### Manifest 主路径

适合:

- 长期训练数据
- 多字段样本
- 多任务数据
- 需要固定 schema 和复现的数据

结论:

- manifest 是长期推荐主路径
- 新模板优先考虑是否能先落到 `Record*`

### 目录分类

适合:

- 标准图片分类
- 现成按类别分目录的数据

结论:

- 已经是目录分类时不必先强制改成 manifest
- 但长期演进到复杂字段后, 仍建议转向 manifest

### Basename sidecar

适合:

- `images/000.png` 对应 `masks/000.png`
- `images/000.png` 对应 `000.txt` / `000.json`

结论:

- 适合作为低门槛接入路径
- 不应把所有 sidecar 变体硬塞进一个超大 dataset 类

### 纯图片目录

适合:

- 推理集
- 无标签采样
- 自监督预处理

### 标准格式标注

适合:

- `COCO`
- `YOLO`
- `VOC`
- keypoint 相关标准格式

结论:

- 这类模板有明确复用价值时再单独沉淀

## 配置侧边界

dataset 接入 YAML 时遵守三条规则:

- dataset 本身继续使用 `target + params`
- dataloader 通过 `dataset: ${train_dataset}` 复用 dataset 配置
- `collate_fn` 需要特殊处理时也使用 `target + params`

也就是说:

- dataset 模板规范定义"该选什么模板"
- 配置工作流页面定义"在 YAML 里怎样接进来"

## 新增 dataset 的判断顺序

1. 只是字段不同, 是否能直接复用现有 `Record*`
2. 只是磁盘对齐方式不同, 是否能复用 sidecar / folder 模板
3. 只是 batch 拼接特殊, 是否只需新增 `collate`
4. 只是样本变换特殊, 是否只需新增 transform
5. 确实是新语义形态, 再新增 dataset 文件并注册

## 新增 dataset 的落地要求

- 在 `xdl/dataset/__init__.py` 中集中注册
- 至少覆盖 `__len__`, `__getitem__`, registry 构建和基本 batch 行为
- 同步更新相关文档:
  - 当前页面
  - [../explanation/dataset-structure.md](../explanation/dataset-structure.md)
  - [api-boundary.md](api-boundary.md)
  - `xdl/dataset/AGENTS.md`

## 当前优先方向

- manifest 继续作为长期推荐主路径
- sidecar / image-folder 模板作为低门槛接入路径
- 优先沉淀成可注册, 可 YAML 构建, 可测试的通用模板
- 避免只在训练脚本里临时实现数据接入

## 相关页面

- [xdl.md](xdl.md)
- [../explanation/dataset-structure.md](../explanation/dataset-structure.md)
- [../usage/xdl-config-workflows.md](../usage/xdl-config-workflows.md)
- [../README.md#xdl-dataset-模板规划](../README.md#xdl-dataset-模板规划)
