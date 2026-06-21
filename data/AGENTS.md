# data — 本地数据目录

## 目录职责

- 存放 XDL 训练,验证和推理所需的本地数据集,样本数据,中间产物和缓存
- 是仓库里所有数据本体的统一落点,不把数据散落在 `datasets/`,`raw/`,`resources/` 等其他目录

## 数据放置规则

- **所有数据集和样本数据一律放进 `data/`**,不新建顶层 `datasets/`,`raw/`,`resources/` 等平行目录.历史上仓库根目录曾有 `datasets/` 存放 `coco8` 样本,现已并入 `data/`,统一归 `data/` 管理.
- 小型内置样本(例如 `coco8`)放在 `data/<name>/`,大型或可重新下载的数据集放在 `data/<dataset_name>/` 下,需要时再分子目录(如 `images/`,`labels/`,`train/`,`val/`).
- 配置和训练脚本里引用本地数据时,优先用 `${xdl.abspath:${xdl.config_dir},data/...}` 或显式相对 `data/` 的路径,不要硬编码绝对路径或用户家目录.
- 不要把模型权重,checkpoint,日志,产物放到 `data/`,这些分别归 `artifacts/`,`outputs/`,`runs/` 等 `.gitignore` 管理的目录.

## 跟踪策略

- `data/` 整体被 `.gitignore` 忽略,数据本体(图片,标注,tar.gz,大文件)一律**不入库**,保持仓库轻量,避免大文件和跨机器不可复现的二进制进入版本库.
- 本目录的 `AGENTS.md` 是**唯一例外**,通过 `.gitignore` 的 `!data/AGENTS.md` 规则保留入库,作为给 agent 和开发者看的目录说明.
- 需要样本数据时,通过下载脚本(如 `tools/dataset/`)或文档说明在本地准备,不要把样本数据 commit 进仓库.

## 当前内容(示例,不入库)

- `MNIST/`,`FashionMNIST/`:torchvision 自动下载的 MNIST 系列数据
- `cifar-100-python.tar.gz`:CIFAR-100 压缩包
- `coco8/`:Ultralytics COCO8 检测样本(8 张图 + YOLO 标签),曾入库的 `labels/`,`README.md`,`LICENSE` 已改为纯本地,不跟踪

## 修改约束

- 新增数据放置约定或目录布局变化时,同步更新本文件,根 `AGENTS.md` 的"数据集存放"约定,以及 `docs/md/README.md#xdl-dataset-模板规划`.
- 不要在本目录提交数据本体;如确需极小的自包含测试样本,放到 `tests/` 并单独评估,而不是放进 `data/`.
