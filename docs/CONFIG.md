# XDL Config 系统说明

本文档只讲 XDL 当前的配置系统，不重复介绍整个框架结构。

## 1. 配置系统解决什么问题

XDL 的 config 系统负责把 YAML 变成可运行组件集合。它解决的是：

- 固定实验配置的上层结构
- 用统一写法构建模型、数据集、优化器、调度器、loss、metrics
- 把配置解析和对象实例化从训练脚本里抽出来

它不负责替代 `Trainer`，也不试图把全部训练逻辑都塞进 YAML。

## 2. 当前主链路

当前推荐入口是：

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
```

主链路如下：

```text
YAML
  -> load_config_with_schema()
  -> to_plain_dict()
  -> build_model / build_dataset / build_dataloader
  -> build_optimizer / build_scheduler / build_loss / build_metrics
  -> TrainSetup
```

返回结果是 `TrainSetup`，其中包含：

- `model`
- `train_loader`
- `val_loader`
- `test_loader`
- `optimizer`
- `scheduler`
- `loss_fn`
- `metrics`
- `logging_config`
- `checkpoint_config`
- `accelerate_config`
- `trainer_config` 以及 `precision` / `gradient_accumulation_steps` 等常用 trainer 字段

如果要直接交给 `Trainer.fit()`，可以继续：

```python
model = setup.create_model()
```

公共入口契约见 [API.md](API.md)。配置主链路推荐只依赖 `from xdl.config import setup_from_yaml, TrainSetup`，不要直接依赖 `xdl.config.setup` 内部辅助函数。

## 3. 模块分工

`xdl/config/` 主要由四部分组成：

- [schema.py](../xdl/config/schema.py)：固定顶层结构和默认值
- [resolver.py](../xdl/config/resolver.py)：schema merge、插值解析、普通 dict 转换
- [builder.py](../xdl/config/builder.py)：组件实例化
- [setup.py](../xdl/config/setup.py)：组装整条构建链路

职责边界如下：

- schema 决定“允许什么结构”
- resolver 决定“配置怎样被解析”
- builder 决定“对象怎样被构建”
- setup 决定“怎样把对象拼成 `TrainSetup`”

## 4. 顶层结构

当前 schema 版本为 `v1`，顶层字段包括：

- `config_version`
- `runtime`
- `trainer`
- `model`
- `task`
- `train_transforms` / `val_transforms` / `test_transforms`
- `train_dataset` / `val_dataset` / `test_dataset`
- `dataloader_defaults`
- `train_dataloader` / `val_dataloader` / `test_dataloader`
- `optimization`
- `loss`
- `metrics`
- `callbacks`
- `logging`
- `checkpoint`
- `accelerate`
- `deepspeed`
- `xdl`

其中最常用的几块是：

- `runtime`：设备、实验名、输出目录
- `trainer`：epoch、batch size、precision、梯度累积
- `model`：模型组件
- `task`: 可选的 `CoreModel` 任务组件, 适合大模型或手写训练逻辑
- `optimization`：优化器和调度器
- `loss` / `metrics`：训练目标和评估指标
- `callbacks`: 用 import path 配置化构建 callback

## 5. 统一组件写法

XDL 当前统一使用 `target + params`：

```yaml
model:
  target: "registry:simple_mlp"
  params:
    input_size: 784
    hidden_size: 128
    num_classes: 10
```

`target` 使用 `source:name` 形式。

常见来源：

- `registry:...`
- `torch.nn:...`
- `torch.optim:...`
- `torch.optim.lr_scheduler:...`
- `torchvision.transforms:...`

`builder.py` 会先解析 `target`，再实例化组件。

内置上下文字段由配置加载器注入:

```yaml
runtime:
  output_dir: ${xdl.abspath:${xdl.config_dir},outputs}
```

可用字段:

- `${xdl.config_path}`: 当前 YAML 文件的绝对路径
- `${xdl.config_dir}`: 当前 YAML 文件所在目录
- `${xdl.project_root}`: 调用 `setup_from_yaml()` 时的当前工作目录
- `${xdl.join_path:...}` / `${xdl.abspath:...}`: 路径拼接和绝对路径 resolver

## 6. transform、dataset、dataloader 的关系

配置系统把数据链路拆成三层：

1. transform
2. dataset
3. dataloader

示例：

```yaml
dataloader_defaults:
  batch_size: ${trainer.batch_size}
  num_workers: 0
  pin_memory: false

train_dataset:
  target: "registry:SyntheticClassificationDataset"
  params:
    num_samples: 500
    input_shape: [784]
    num_classes: 10
    seed: 42

train_dataloader:
  dataset: ${train_dataset}
  params:
    shuffle: true
    drop_last: true
```

当前实现里：

- `dataloader_defaults` 负责公共默认值
- `train_dataloader.params` 负责局部覆盖
- `val_dataloader.dataset: ${train_dataset}` 这类写法可以复用已有 dataset 配置
- `trainer.batch_size` 可作为默认 batch size 来源
- `collate_fn` 支持 `None`、可调用对象或 `target + params`

完整 dataset 模板规划, 选择表和新增 dataset 流程见 [DATASET.md](DATASET.md). 本节只保留配置系统需要知道的写法.

### 先区分两种“数据类型”

规划 dataset 模板时，先区分两个维度：

1. 样本语义形态：单条样本里到底有哪些字段。
2. 磁盘组织形态：这些字段在目录、sidecar 文件或集中标注里怎样存放。

这两个维度不要混在一起。比如：

- `image + text` 是样本语义形态
- `000.png` 对应 `000.txt` 是磁盘组织形态

同一个语义形态可以有多种组织方式；同一种组织方式也可以承载不同任务。

### 样本语义形态

当前 XDL 已覆盖或正在补齐的主流语义形态包括：

- `image + label`：单标签分类
- `image + target`：回归
- `image + labels`：多标签分类
- `image + text`：图文配对或图像生成微调
- `text (+ target_text)`：纯文本或 instruction / response
- `image_a + image_b (+ label)`：pair / matching / contrastive
- `anchor + positive + negative`：triplet / retrieval
- `image + mask`：分割等 dense target
- `image + boxes + labels`：检测
- `source_image + target_image (+ reference_image + edit_mask + prompt)`：编辑 / 条件生成

### 磁盘组织形态

仓库当前已经出现, 或后续非常值得支持的组织方式主要有：

1. 目录分类型

```text
root/
  cat/
    000.png
  dog/
    001.png
```

- 标签来自父目录名
- 对应 `image + label`
- 当前模板：`ImageFolderClassificationDataset`

2. Manifest 显式字段型

```jsonl
{"image": "images/000.png", "prompt": "a studio portrait"}
{"image": "images/001.png", "label": "cat"}
```

- 字段最明确, 可扩展性最好
- 当前是 XDL 推荐主路径
- 当前模板覆盖 classification / regression / multilabel / image-text / text / pair / triplet / segmentation / detection / image-edit

3. Basename 对齐的 sidecar 型

```text
data/
  000.png
  000.txt
  001.png
  001.txt
```

或：

```text
images/
  000.png
texts/
  000.txt
```

- 通过同名 stem 对齐
- 常见于 `image + text`, `image + label`, `image + mask`
- 仓库里 `train/train_sd35m_apex_xdl.py` 已经有 `image.with_suffix(".txt")` 的手写实现
- 当前模板: `ImageTextSidecarDataset`, `ImageMaskSidecarDataset`
- 后续 `image + json` / `image + label` 仍建议继续补 basename 对齐模板

4. 多目录对齐型

```text
images/
  000.png
masks/
  000.png
```

- 本质上仍是 basename 对齐
- 常见于分割、depth、编辑、多模态条件输入
- 当前模板: `ImageMaskSidecarDataset`; `ImageEditDataset` 的真实样本可以由 manifest 描述这类结构

5. 纯图片目录

```text
images/
  000.png
  001.png
```

- 没有标签或说明文件
- 常见于推理、生成模型无监督图像源、自监督、特征提取
- 当前模板：`ImageFolderDataset`, 返回 `image + sample_id + image_path`

6. 集中标注文件型

- 如 COCO / YOLO / VOC / keypoint json
- 当前 examples 还没有这类官方样例
- 但这是后续生态兼容的重要方向

常用内置 dataset 模板:

- `registry:ImageFolderDataset`: 读取纯图片目录, 返回 `image + sample_id`, 可选 `image_path`.
- `registry:ImageFolderClassificationDataset`: 读取 `root/class_name/image` 分类目录.
- `registry:RecordClassificationDataset`: 从 JSONL/JSON/CSV manifest 读取 `image + label`.
- `registry:RecordRegressionDataset`: 从 manifest 读取 `image + target`, 适合分数、年龄、质量估计等回归任务.
- `registry:RecordMultiLabelClassificationDataset`: 从 manifest 读取 `image + labels`, 适合多标签分类.
- `registry:ImageMaskSidecarDataset`: 从同名 mask sidecar 读取 `image + mask`, 支持同目录或 image/mask 分目录.
- `registry:RecordSegmentationDataset`: 从 manifest 读取 `image + mask`, 适合语义分割和其他 dense label 任务.
- `registry:RecordDetectionDataset`: 从 manifest 读取 `image + boxes + labels`, 适合目标检测和变长 target 任务.
- `registry:RecordImageTextDataset`: 从 manifest 读取 `image + text/prompt/caption`, 适合图文微调.
- `registry:ImageTextSidecarDataset`: 从同名 `.txt` sidecar 读取 `image + text`, 支持同目录或 image/text 分目录.
- `registry:RecordTextDataset`: 从 manifest 读取 `text/prompt/caption`, 可选 `target_text/response/completion`, 适合纯文本或指令样本.
- `registry:RecordPairDataset`: 从 manifest 读取 `image_a + image_b`, 可选 `label` / `text_a` / `text_b`, 适合 siamese、对比学习、检索 pair.
- `registry:RecordTripletDataset`: 从 manifest 读取 `anchor/positive/negative` 图像三元组, 可选各自文本字段, 适合 metric learning 和 retrieval.
- `registry:ImageEditDataset`: 从 manifest 读取 source/target/reference/mask 多图编辑样本.
- `registry:RecordDataset`: 直接返回 manifest 里的 dict 记录.

历史 `registry:Manifest*` 名称继续可用, 但新配置优先写 `registry:Record*` 或 `registry:ImageEditDataset` / `registry:ImageEditCollate`.

### Manifest 模板字段约定

这些模板优先复用一组固定字段名, 这样 YAML 和下游 `CoreModel` 更容易保持一致:

- 分类: `image`, `label`
- 回归: `image`, `target`
- 多标签: `image`, `labels`
- 分割: `image`, `mask`
- 检测: `image`, `boxes`, `labels`
- 图文: `image`, `text` 或 `prompt` / `caption`
- 文本: `text` 或 `prompt` / `caption`, 可选 `target_text` / `response` / `completion`
- Pair: `image_a`, `image_b`, 可选 `label`, `text_a`, `text_b`
- Triplet: `anchor_image`, `positive_image`, `negative_image`, 可选 `anchor_text`, `positive_text`, `negative_text`
- Image edit: `source_image`, `target_image`, 可选 `reference_image`, `edit_mask`, `prompt`
- 通用样本标识: `sample_id` 或 `id`

如果 manifest 使用不同字段名, 优先通过 dataset 参数显式映射, 不要在训练入口里写额外字段迁移逻辑.

### 按数据来规划 dataset

实际落模板时，优先先看“手里的数据长什么样”，再决定 dataset 类型。

推荐的判断顺序：

1. 先看磁盘组织形态
   - 目录分类型：优先 `ImageFolderClassificationDataset`
   - manifest 明确字段: 优先 `Record*` 模板
   - `000.png + 000.txt`：优先 `ImageTextSidecarDataset`
   - `images/000.png + masks/000.png`: 优先 `ImageMaskSidecarDataset`
   - `000.png + 000.json`: 后续走 basename 对齐模板或薄 manifest adapter
   - 只有图片：优先 `ImageFolderDataset`
2. 再看样本语义形态
   - 分类 / 回归 / 多标签
   - 图文 / 文本
   - pair / triplet
   - dense / detection / edit
3. 最后再决定 transform 和 collate
   - 单输入任务通常可直接复用普通 transform
   - `image + mask` / `image + boxes` / 多图编辑样本需要同步 transform
   - 变长 target 任务要单独 collate

可以把当前和后续的 dataset 规划概括成 4 组：

1. Manifest 主路径

- 适合长期可维护的数据
- 适合多字段、多任务、可扩展 schema
- 当前优先级最高

2. 目录快捷路径

- 适合最常见的目录分类和纯图片输入
- 目标是减少写 manifest 的门槛
- 当前已有 `ImageFolderClassificationDataset` 和 `ImageFolderDataset`

3. Sidecar / basename 对齐路径

- 适合 `000.png` 对应 `000.txt` / `000.json` / `000.png`
- 适合 diffusion 微调、caption、小型多模态数据、分割 mask
- 当前已有 `ImageTextSidecarDataset` 覆盖 `image + txt`
- 当前已有 `ImageMaskSidecarDataset` 覆盖 `image + mask`
- 后续建议继续补更通用的 `BasenameAlignedDataset`

4. 标准格式适配路径

- 适合 COCO / YOLO / VOC / keypoint 等通用生态数据
- 目标是减少用户自己写 parser
- 后续建议逐步补：
  - `COCODetectionDataset`
  - `COCOSegmentationDataset`
  - `RecordKeypointDataset` 或标准 keypoint 适配模板

### Dataset 样例

纯图片目录:

```yaml
train_dataset:
  target: "registry:ImageFolderDataset"
  params:
    root: ${xdl.abspath:${xdl.config_dir},data/images}
    transform: ${train_transforms}
    recursive: true
```

分类:

```yaml
train_dataset:
  target: "registry:RecordClassificationDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_cls.jsonl}
    transform: ${train_transforms}
```

分割:

```yaml
train_dataset:
  target: "registry:RecordSegmentationDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_seg.jsonl}
    transform:
      target: "registry:ImageMaskTransform"
      params:
        height: 512
        width: 512
        random_flip: true
```

image-mask sidecar:

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

检测:

```yaml
train_dataset:
  target: "registry:RecordDetectionDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_det.jsonl}
    transform:
      target: "registry:ImageBoxesTransform"
      params:
        height: 640
        width: 640

train_dataloader:
  dataset: ${train_dataset}
  collate_fn:
    target: "registry:DetectionCollate"
  params:
    batch_size: 4
    shuffle: true
```

image-text:

```yaml
train_dataset:
  target: "registry:RecordImageTextDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_it.jsonl}
    transform: ${train_transforms}
    text_keys: ["prompt", "caption", "text"]
```

image-text sidecar:

```yaml
train_dataset:
  target: "registry:ImageTextSidecarDataset"
  params:
    root: ${xdl.abspath:${xdl.config_dir},data/image_text}
    transform: ${train_transforms}
    text_extension: ".txt"
    text_selection: "first_line"
    missing_text: "skip"
```

text-only:

```yaml
train_dataset:
  target: "registry:RecordTextDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_text.jsonl}
    text_keys: ["prompt", "text"]
    target_text_keys: ["response", "target_text"]
```

triplet:

```yaml
train_dataset:
  target: "registry:RecordTripletDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_triplet.jsonl}
    transform: ${train_transforms}
    include_paths: true
```

image edit:

```yaml
train_dataset:
  target: "registry:ImageEditDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_edit.jsonl}
    height: 512
    width: 512
    random_flip: true

train_dataloader:
  dataset: ${train_dataset}
  collate_fn:
    target: "registry:ImageEditCollate"
  params:
    batch_size: 2
    shuffle: true
```

## 7. loss 和 metrics 的写法

### 单个 loss

```yaml
loss:
  - target: "torch.nn:CrossEntropyLoss"
    params: {}
```

### 多个 loss

`build_loss()` 支持列表形式，多项时会构建 `WeightedLoss`：

```yaml
loss:
  - target: "torch.nn:MSELoss"
    params: {}
    weight: 1.0
  - target: "registry:DiceLoss"
    params: {}
    weight: 0.5
```

### metrics

```yaml
metrics:
  - target: "registry:Accuracy"
    params:
      num_classes: 10
```

`build_metrics()` 返回指标实例列表。

### callbacks

Callback 暂不新增 registry 类型, 使用 import path 构建:

```yaml
callbacks:
  - target: "xdl.callbacks:SaveTrainableStateCallback"
    params:
      dirpath: ${xdl.abspath:${xdl.config_dir},adapters}
      every_n_epochs: 1
```

`Trainer.from_setup(setup)` 会把这些 callback 加入训练器。

### dataset 参数中的可调用 transform

除了 `transform` / `target_transform`, 内置 manifest 文本模板还支持把
`text_transform` 和 `target_text_transform` 写成 `target + params` 组件配置:

```yaml
train_dataset:
  target: "registry:RecordTextDataset"
  params:
    manifest_path: ${xdl.abspath:${xdl.config_dir},data/train_text.jsonl}
    text_transform:
      target: "torch.nn:Identity"
      params: {}
```

### CoreModel task

标准监督任务继续使用 `model + optimization + loss`. 如果任务本身继承
`CoreModel` 并在 `configure_optimizers()` 中构建优化器, 可以改用:

```yaml
task:
  target: "my_project.tasks:MyTask"
  params:
    lr: 0.0001
```

此时 `optimizer` 和 `loss` 可以省略, `setup.create_model()` 会直接返回该
`CoreModel` 实例。

## 8. 一个最小可运行示例

仓库里的 [config/unified_logger_example.yaml](../config/unified_logger_example.yaml) 是当前最合适的主路径样例。

如果想看 manifest 数据模板的完整官方样例, 参考:

- [config/manifest_segmentation_example.yaml](../config/manifest_segmentation_example.yaml)
- [config/manifest_detection_example.yaml](../config/manifest_detection_example.yaml)
- [config/manifest_regression_example.yaml](../config/manifest_regression_example.yaml)
- [config/manifest_pair_example.yaml](../config/manifest_pair_example.yaml)

最小消费方式：

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")

print(type(setup.model).__name__)
print(type(setup.optimizer).__name__)
print(type(setup.loss_fn).__name__)
print(len(setup.train_loader.dataset))
```

## 9. 推荐扩展方式

当你想扩展配置系统时，先判断变化属于哪类：

- 新顶层字段：改 `schema.py`
- 新插值或 merge 规则：改 `resolver.py`
- 新组件构建方式：改 `builder.py`
- 新装配流程：改 `setup.py`

不要把所有变化都堆到 `setup_from_yaml()`。

训练入口里的轻量 dataclass 配置推荐使用:

```python
from xdl.config import load_structured_dataclass_config

config = load_structured_dataclass_config(MyConfig, "train.yaml")
```

该工具遵循 `structured dataclass 默认值 -> YAML 覆盖 -> overrides 覆盖`
的顺序, 并统一走 OmegaConf resolver 和插值解析。

## 10. 当前边界

当前 config 系统已经稳定支持：

- schema v1 顶层结构
- `target + params`
- transform / dataset / dataloader 构建
- optimizer / scheduler / loss / metrics 构建
- `TrainSetup` 返回

但仍有边界：

- 它不替代训练主循环
- 它不定义统一 CLI
- 它不自动覆盖所有任务特化逻辑
- 它更适合“配置化构建组件”，而不是“声明式描述整个实验世界”
- schema dataclass、resolver 和 builder 内部辅助函数仍属于演进中的 API；稳定入口以 [API.md](API.md) 为准
