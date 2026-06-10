# xdl/dataset — 数据集子模块

## 目录职责

- 定义数据集、transform 与 collate 相关实现
- 通过注册系统为配置化训练提供数据输入能力

## 当前内容

- `vision_datasets.py`：常用视觉数据集
- `basic.py`：基础或合成数据集
- `hairdata*.py`：毛发相关数据集
- `hair_transforms.py`：相关 transform
- `manifest_image_edit.py`: 通用多图 image edit manifest 数据集和 paired transform
- `templates.py`: 通用 manifest, image folder, image-text 数据集模板
- `collate.py`：批处理拼接逻辑

## 修改约束

- 数据集、transform、collate 变更要考虑 registry 注册方式
- batch 结构尽量清晰，复杂嵌套返回要同步考虑 Trainer 设备迁移限制
- 路径、标注格式、可选依赖要写清楚
- 可选依赖应优雅降级，不要让整个包在导入时失败

## 验证建议

- 修改后优先看 `tests/dataset/`
- 新数据集至少验证 `__len__`、`__getitem__` 和基本 batch 流程
