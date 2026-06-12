# xdl/dataset — 数据集子模块

## 目录职责

- 定义数据集、transform 与 collate 相关实现
- 通过注册系统为配置化训练提供数据输入能力

## 当前内容

- `utils.py`: 路径操作, manifest 读盘, record 解析, sidecar 对齐, tensor 转换等公共 helper
- `collate.py`: 通用 collate 和任务 collate
- `transforms.py`: 需要同步处理 image / mask / boxes / 多图样本的 transform
- `basic.py`: 基础或合成数据集
- `vision.py`: 常用 torchvision 数据集封装
- `record.py`: manifest record 基类和通用 record dataset
- `folder.py`: 纯图片目录 dataset
- `sidecar.py`: 通用 basename sidecar 对齐 dataset
- `classification.py`: 单标签和多标签分类 dataset
- `regression.py`: 回归 dataset
- `segmentation.py`: image + mask 分割 dataset
- `detection.py`: image + boxes + labels 检测 dataset
- `image_text.py`: image + text dataset
- `text.py`: 文本 record dataset
- `pair.py`: pair / siamese / contrastive dataset
- `triplet.py`: triplet / retrieval dataset
- `image_edit.py`: 图像编辑 dataset
- `split.py`: dataset 拆分 helper
- `hair/`: 毛发相关数据集和 transform

## 推荐模板

- `RecordDataset`：最通用的 manifest dict 读取模板
- `ImageFolderDataset`：纯图片目录模板, 适合推理、自监督或无标签图像源
- `ImageFolderClassificationDataset`：最短路径的目录分类模板
- `ImageTextSidecarDataset`：`000.png` 对应 `000.txt` 的图文 sidecar 模板
- `ImageMaskSidecarDataset`: `images/000.png` 对应 `masks/000.png` 的 image-mask sidecar 模板
- `RecordClassificationDataset`：真实项目更常见的 manifest 分类模板
- `RecordRegressionDataset`：分数, 年龄, 质量估计等回归模板
- `RecordMultiLabelClassificationDataset`：多标签分类模板
- `RecordSegmentationDataset`：`image + mask` 分割 / 逐像素标注模板
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

当前主路径是 `Record*` / `ImageEdit*` 模板读取 manifest 文件. 历史 `Manifest*` 类名和 registry 键继续保留为兼容别名, 新代码优先使用 `Record*` / `ImageEdit*` 名称.

仓库里已经存在的真实数据组织方式包括：

- 目录分类：`root/class_name/image`
- 纯图片目录：`root/image`
- manifest 显式字段：classification / regression / segmentation / detection / image-text / image-edit
- 多图编辑样本：`source_image + target_image + reference_image + mask`
- `image + txt` sidecar：同目录或 image/text 分目录的 basename 对齐
- `image + mask` sidecar: 同目录不同扩展名或 image/mask 分目录的 basename 对齐

后续优先补的组织形态模板：

- ~~更通用的 `BasenameAlignedDataset`, 覆盖 `image + json` / `image + label` 等 sidecar 结构~~ 已实现
- 标准格式适配：COCO / keypoint

## 新增：BasenameAlignedDataset

`BasenameAlignedDataset` 是一个通用的 basename 对齐数据集模板, 替代为每种 sidecar 格式写新类的模式.

- 通过 `sidecar_extension` 指定 sidecar 文件扩展名
- 内置 reader: `.txt/.text` (字符串), `.json` (解析后的 dict/list), 图片扩展名 (PIL.Image)
- 支持 `sidecar_reader` 自定义读取逻辑, `sidecar_transform` 对 sidecar 内容做变换
- `transform` 仅作用于 image, 不做 paired transform (需要 paired transform 的场景继续用 `ImageMaskSidecarDataset`)

## 新增：split_dataset / train_val_split

轻量数据集拆分工具, 封装 torch 的 `random_split`:

- `split_dataset(dataset, lengths, *, seed=42)` — 确定性随机拆分, 返回 `list[Subset]`
- `train_val_split(dataset, val_ratio=0.1, *, seed=42)` — 拆分为训练集和验证集

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
