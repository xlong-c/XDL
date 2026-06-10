# xdl/dataset — 数据集子模块

## 目录职责

- 定义数据集、transform 与 collate 相关实现
- 通过注册系统为配置化训练提供数据输入能力

## 当前内容

- `vision_datasets.py`：常用视觉数据集
- `basic.py`：基础或合成数据集
- `hairdata*.py`：毛发相关数据集
- `hair_transforms.py`：相关 transform
- `_manifest.py`：manifest 读盘与路径解析公共 helper
- `manifest_image_edit.py`: 通用多图 image edit manifest 数据集和 paired transform
- `manifest_dense.py`: 分割, 检测数据集模板, 以及 image-mask / image-boxes transform
- `templates.py`: 通用 manifest, image folder, classification, regression, image-text, pair 数据集模板
- `collate.py`：通用批处理拼接逻辑

## 推荐模板

- `ManifestRecordDataset`：最通用的 manifest dict 读取模板
- `ImageFolderClassificationDataset`：最短路径的目录分类模板
- `ManifestClassificationDataset`：真实项目更常见的 manifest 分类模板
- `ManifestRegressionDataset`：分数, 年龄, 质量估计等回归模板
- `ManifestSegmentationDataset`：`image + mask` dense prediction 模板
- `ManifestDetectionDataset`：`image + boxes + labels` 检测模板
- `ManifestImageTextDataset`：图文配对模板
- `ManifestPairDataset`：siamese / contrastive / retrieval pair 模板
- `ManifestImageEditDataset`：source/target/reference/mask 编辑模板

## 推荐字段名

- 分类: `image`, `label`
- 回归: `image`, `target`
- 分割: `image`, `mask`
- 检测: `image`, `boxes`, `labels`
- 图文: `image`, `text` 或 `prompt` / `caption`
- Pair: `image_a`, `image_b`, 可选 `label`, `text_a`, `text_b`
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
