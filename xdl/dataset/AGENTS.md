# xdl/dataset — 数据集子模块

## 目录职责

- 定义数据集、transform 与 collate 相关实现
- 通过注册系统为配置化训练提供数据输入能力

## 当前内容

- `vision_datasets.py`：常用视觉数据集
- `basic.py`：基础或合成数据集
- `hairdata*.py`：毛发相关数据集
- `hair_transforms.py`：相关 transform
- `_records.py`：manifest 读盘与路径解析公共 helper
- `_paths.py`: 图片扫描, 扩展名归一化, basename sidecar 对齐和 path sample_id helper
- `image_edit.py`: 通用多图 image edit manifest 数据集和 paired transform
- `dense.py`: 分割, 检测, image-mask sidecar 数据集模板, 以及 image-mask / image-boxes transform
- `templates.py`: 通用 manifest, image folder, image-only, image-text sidecar, classification, regression, multi-label, image-text, text, pair, triplet 数据集模板
- `collate.py`：通用批处理拼接逻辑

## 推荐模板

- `RecordDataset`：最通用的 manifest dict 读取模板
- `ImageFolderDataset`：纯图片目录模板, 适合推理、自监督或无标签图像源
- `ImageFolderClassificationDataset`：最短路径的目录分类模板
- `ImageTextSidecarDataset`：`000.png` 对应 `000.txt` 的图文 sidecar 模板
- `ImageMaskSidecarDataset`: `images/000.png` 对应 `masks/000.png` 的 image-mask sidecar 模板
- `RecordClassificationDataset`：真实项目更常见的 manifest 分类模板
- `RecordRegressionDataset`：分数, 年龄, 质量估计等回归模板
- `RecordMultiLabelClassificationDataset`：多标签分类模板
- `RecordSegmentationDataset`：`image + mask` dense prediction 模板
- `RecordDetectionDataset`：`image + boxes + labels` 检测模板
- `RecordImageTextDataset`：图文配对模板
- `RecordTextDataset`：纯文本 / 指令 / 响应样本模板
- `RecordPairDataset`：siamese / contrastive / retrieval pair 模板
- `RecordTripletDataset`：anchor / positive / negative 检索模板
- `ImageEditDataset`：source/target/reference/mask 编辑模板

## 数据组织形态

当前和后续规划中，优先区分两层：

1. 样本语义形态：`image + label`、`image + text`、`pair`、`triplet`、`image + mask` 等
2. 磁盘组织形态：manifest、目录分类、basename sidecar 对齐、纯图片目录、标准格式标注

当前主路径是 `Record*` 模板读取 manifest 文件. 历史 `Manifest*` 类名和 registry 键继续保留为兼容别名, 新代码优先使用 `Record*` / `ImageEdit*` 名称.

仓库里已经存在的真实数据组织方式包括：

- 目录分类：`root/class_name/image`
- 纯图片目录：`root/image`
- manifest 显式字段：classification / regression / segmentation / detection / image-text / image-edit
- 多图编辑样本：`source_image + target_image + reference_image + mask`
- `image + txt` sidecar：同目录或 image/text 分目录的 basename 对齐
- `image + mask` sidecar: 同目录不同扩展名或 image/mask 分目录的 basename 对齐

后续优先补的组织形态模板：

- 更通用的 `BasenameAlignedDataset`, 覆盖 `image + json` / `image + label` 等 sidecar 结构
- 标准格式适配：COCO / keypoint

## 推荐字段名

- 分类: `image`, `label`
- 回归: `image`, `target`
- 多标签: `image`, `labels`
- 分割: `image`, `mask`
- 检测: `image`, `boxes`, `labels`
- 图文: `image`, `text` 或 `prompt` / `caption`
- 文本: `text` 或 `prompt` / `caption`, 可选 `target_text` / `response` / `completion`
- Pair: `image_a`, `image_b`, 可选 `label`, `text_a`, `text_b`
- Triplet: `anchor_image`, `positive_image`, `negative_image`, 可选 `anchor_text`, `positive_text`, `negative_text`
- 编辑: `source_image`, `target_image`, 可选 `reference_image`, `edit_mask`, `prompt`
- 通用样本标识: `sample_id` 或 `id`

## 修改约束

- 数据集、transform、collate 变更要考虑 registry 注册方式
- batch 结构尽量清晰，复杂嵌套返回要同步考虑 Trainer 设备迁移限制
- 路径、标注格式、可选依赖要写清楚
- 可选依赖应优雅降级，不要让整个包在导入时失败

## 验证建议

- 修改后优先看 `tests/dataset/`
- 新数据集至少验证 `__len__`、`__getitem__` 和基本 batch 流程
