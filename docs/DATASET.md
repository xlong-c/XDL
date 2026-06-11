# XDL Dataset 模板规划

本文档负责说明 XDL 数据集模板怎样选择, 怎样扩展, 以及新增数据集时优先复用哪些基类和模板. 配置系统写法只保留必要示例, 完整 `target + params` 规则见 [CONFIG.md](CONFIG.md).

## 1. 规划原则

设计 dataset 时先分清两个维度:

- 样本语义形态: 单条样本返回哪些字段, 例如 `image + label`, `image + mask`, `image + text`, `pair`, `triplet`, `text + target_text`.
- 磁盘组织形态: 数据在磁盘上怎样摆放, 例如 manifest, 目录分类, basename sidecar, 纯图片目录, 标准标注格式.

XDL 的推荐顺序是:

1. 能写 manifest 时, 优先使用 `Record*` 模板. 字段清晰, 可扩展, 容易跨机器复现.
2. 数据已经是目录分类或纯图片目录时, 使用目录模板, 不强制先生成 manifest.
3. 数据是同名文件对齐时, 使用 sidecar 模板, 例如 `images/000.png` 对应 `masks/000.png` 或 `000.txt`.
4. 只有当字段解析, 采样逻辑或外部格式确实特殊时, 才新增薄 dataset 类.
5. 新增模板必须通过 `xdl.dataset.__init__` 集中注册, 并至少覆盖 `__len__`, `__getitem__`, registry 构建和基本 batch 行为测试.

命名约定: 新代码优先使用 `Record*` / `ImageEdit*` dataset 名称. 历史 `Manifest*` 类名和 registry 键仍保留为兼容别名, 例如 `ManifestRegressionDataset` 仍等价于 `RecordRegressionDataset`.

## 2. 样本语义形态

| 语义形态 | 典型任务 | 推荐模板 |
|---|---|---|
| `image` | 推理, 自监督, 特征提取 | `ImageFolderDataset` |
| `image + label` | 单标签分类 | `ImageFolderClassificationDataset`, `RecordClassificationDataset` |
| `image + target` | 回归, 质量估计, 年龄预测 | `RecordRegressionDataset` |
| `image + labels` | 多标签分类 | `RecordMultiLabelClassificationDataset` |
| `image + mask` | 语义分割, depth, dense label | `RecordSegmentationDataset`, `ImageMaskSidecarDataset` |
| `image + boxes + labels` | 目标检测 | `RecordDetectionDataset` |
| `image + text` | caption, diffusion 微调, 图文检索 | `RecordImageTextDataset`, `ImageTextSidecarDataset` |
| `text (+ target_text)` | 文本分类前处理, SFT, 指令数据 | `RecordTextDataset` |
| `image_a + image_b (+ label)` | pair matching, siamese, 对比学习 | `RecordPairDataset` |
| `anchor + positive + negative` | metric learning, retrieval | `RecordTripletDataset` |
| `source_image + target_image (+ reference_image + edit_mask + prompt)` | 图像编辑, 条件生成 | `ImageEditDataset` |
| 任意 dict record | 表格, 推荐, 时序窗口, 私有 schema | `RecordDataset` 或继承 `RecordDatasetBase` |

## 3. 磁盘组织形态

### Manifest 主路径

Manifest 支持 `.jsonl`, `.json`, `.csv`, 相对路径默认基于 manifest 所在目录解析.

```json
{"image": "images/000.png", "label": "cat", "sample_id": "000"}
{"image": "images/001.png", "label": "dog", "sample_id": "001"}
```

适合:

- 长期训练数据
- 多字段样本
- 多任务数据
- 分布式训练前的数据快照
- 需要固定 schema 和复现的数据

新增 manifest 数据集时优先继承 `RecordDatasetBase`, 复用:

- `load_manifest_context()`
- `self.records`
- `self.base_dir`
- `self._record_at(index)`
- `self._resolve_record_path(record, key)`
- `self._sample_id_from_path(...)`
- `self._sample_id_from_fallback(...)`

### 目录分类

```text
root/
  cat/
    000.png
  dog/
    001.png
```

使用 `ImageFolderClassificationDataset`, 返回 `(image, target)`.

### 纯图片目录

```text
images/
  000.png
  nested/001.jpg
```

使用 `ImageFolderDataset`, 返回:

```python
{
    "image": image,
    "sample_id": "000",
    "image_path": "...",
}
```

### Image + text sidecar

```text
data/
  000.png
  000.txt
```

或:

```text
images/
  000.png
texts/
  000.txt
```

使用 `ImageTextSidecarDataset`, 返回 `image`, `text`, `sample_id`, 可选路径字段.

### Image + mask sidecar

```text
images/
  000.png
masks/
  000.png
```

或同目录不同扩展名:

```text
data/
  000.jpg
  000.png
```

使用 `ImageMaskSidecarDataset`, 返回 `image`, `mask`, `sample_id`, 可选路径字段. 它与 `RecordSegmentationDataset` 返回结构一致, 因此下游 segmentation task 可以少改或不改.

同目录 sidecar 模式建议显式设置 `extensions`, 用来只扫描 image 文件, 避免把 mask 文件再次当作 image 样本.

### 标准格式适配

COCO, YOLO, VOC, keypoint 等标准格式目前建议先转换为 manifest. 后续如果某个格式在多个项目中反复出现, 再新增专门 dataset 适配器, 但仍应把返回字段对齐到 `image + boxes + labels`, `image + mask`, 或 keypoint 的稳定 dict 结构.

## 4. 加载模式

| 加载模式 | XDL 推荐用法 | 适用场景 |
|---|---|---|
| Map-style dataset | 当前所有内置模板主路径 | 文件数量明确, 可随机访问 |
| Iterable dataset | 暂不作为内置主模板 | 流式日志, 超大文本, remote shard |
| 懒加载 | 图像, mask, text sidecar 模板默认采用 | 大多数图片, 分割, 检测数据 |
| 全量内存 | 自定义 dataset 或 transform 内处理 | 小型表格, toy data |
| 预处理缓存 | 先生成 `.pt`, `.npy`, `.parquet`, `.jsonl` manifest | tokenizer 结果, latent cache, embedding cache |
| 动态 transform | dataset 参数传 `transform` | 图像增强, 同步 mask/box 变换 |
| 自定义 collate | `collate_fn.target: registry:...` | detection, text padding, 复杂 dict batch |
| 分布式采样 | `DataLoader`/训练入口层处理 | 多卡训练, 大规模数据 |

当前内置模板重点覆盖 map-style + 懒加载 + manifest/sidecar/目录组织. Iterable/shard/远程读取可以通过继承 `torch.utils.data.IterableDataset` 单独实现, 但不要把流式逻辑硬塞进 manifest 模板.

## 5. Transform 和 Collate

单输入图像任务可以直接使用普通 image transform.

`image + mask` 使用 `ImageMaskTransform`, 它会对 image 和 mask 共享 resize/crop/flip 决策, 并保证 mask 用 nearest 语义.

`image + boxes` 使用 `ImageBoxesTransform`, 它会同步 resize/flip boxes.

`DetectionCollate` 保留每张图不同数量的 `boxes` 和 `labels`.

`DictCollate` 适合大多数 dict 样本, tensor 字段会尝试 stack, stack 失败时保留 list.

`PadCollate` 适合简单变长序列 padding. 复杂 NLP token padding 更推荐在项目侧写专门 collate 或 tokenizer wrapper.

## 6. YAML 示例

Manifest 分类:

```yaml
train_dataset:
  target: "registry:RecordClassificationDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_cls.jsonl}
    transform: ${train_transforms}
```

Image + mask sidecar:

```yaml
train_dataset:
  target: "registry:ImageMaskSidecarDataset"
  params:
    image_root: ${xdl.abspath:${xdl.config_dir},data/images}
    mask_root: ${xdl.abspath:${xdl.config_dir},data/masks}
    transform:
      target: "registry:ImageMaskTransform"
      params:
        height: 512
        width: 512
        random_flip: true
```

同目录不同扩展名 mask:

```yaml
train_dataset:
  target: "registry:ImageMaskSidecarDataset"
  params:
    root: ${xdl.abspath:${xdl.config_dir},data/image_mask}
    extensions: [".jpg"]
    mask_extension: ".png"
    transform:
      target: "registry:ImageMaskTransform"
      params:
        height: 512
        width: 512
```

Detection 需要自定义 collate:

```yaml
train_dataloader:
  dataset: ${train_dataset}
  collate_fn:
    target: "registry:DetectionCollate"
  params:
    batch_size: 4
    shuffle: true
```

## 7. 新增数据集流程

新增数据集时按下面顺序判断:

1. 字段已经能用 manifest 表达: 直接用现有 manifest 模板, 或继承 `RecordDatasetBase` 只实现 `__getitem__`.
2. 只是路径组织不同: 优先补 sidecar/目录扫描 helper, 不要复制完整 dataset 类.
3. 只是 batch 方式不同: 新增 collate, 不要改 dataset 返回结构.
4. 只是单样本增强不同: 新增 transform, 不要新增 dataset.
5. 确实是新语义形态: 新增 dataset 文件, 在 `xdl/dataset/__init__.py` 注册, 更新 `docs/DATASET.md`, `docs/API.md`, `xdl/dataset/AGENTS.md`, 并补测试.

薄 subclass 示例:

```python
from typing import Any, Dict

from xdl.dataset import RecordDatasetBase


class MyImageJsonDataset(RecordDatasetBase):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        base_index, record = self._record_at(index)
        image_path = self._resolve_record_path(record, "image")
        json_path = self._resolve_record_path(record, "annotation")
        return {
            "image_path": str(image_path),
            "annotation_path": str(json_path),
            "sample_id": self._sample_id_from_path(record, image_path, base_index),
        }
```

如果这个类要进入 XDL 内置组件, 还需要在 `xdl/dataset/__init__.py` 中集中注册:

```python
register_dataset("MyImageJsonDataset")(MyImageJsonDataset)
```

## 8. 验证清单

新增或改动 dataset 后至少验证:

- `len(dataset)` 与样本数量一致.
- `dataset[0]` 返回字段名和类型稳定.
- 相对路径基于 manifest 或 root 正确解析.
- 缺失必需字段或 sidecar 文件时错误清楚.
- `DataLoader(dataset, batch_size=2)` 或指定 collate 能正常产出 batch.
- registry 路径 `build_dataset({"target": "registry:Name", "params": ...})` 可构建.
- 文档中的 YAML 字段名和构造参数一致.
