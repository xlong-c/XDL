# train - 训练入口目录

## 目录职责

- 按训练阶段收拢可运行的训练入口, 划分预训练与后训练两条链路
- `core/` 是入库维护的核心区 (git 跟踪); `pretrain/` 与 `posttrain/` 是本地实验区
  (整体由根 `.gitignore` 忽略), 新增实验文件放本地实验区即可, 无需改 `.gitignore`
- 每个入口的运行 YAML 与脚本同目录放置; `config/` 只放框架级配置样例

## 目录结构

- `core/pretrain/`: 从零训练参考入口 (分类, VAE, GAN, 生成模型预训练等)
- `core/posttrain/`: 后训练参考入口 (GRPO 等) 与有 tests/docs 引用的资产
- `pretrain/`, `posttrain/`: 本地实验脚本与 YAML (忽略, 不入库)

## 划分规则

- 预训练: 从零得到一个 base checkpoint, 数据与任务自包含
- 后训练: 输入是已有 checkpoint (冻结主干 + 可训练 adapter 或全量微调),
  产出的 adapter/merge 结果交由外部推理或压缩工具链使用
- 新增入口先判断阶段归属再选目录: 需要入库的入口放 `core/` 对应阶段子目录,
  仅本地使用的实验放 `pretrain/` 或 `posttrain/`; 两阶段通用的共享逻辑下沉到 `xdl/`
  (后训练组件在 `xdl/post_training/`), 不要在入口之间互相 import

## 修改约束

- 不引入命令行参数解析库, 配置走 YAML 或代码内显式配置
- 入口保持薄, 优先调用 `xdl/trainer` 生命周期与 `xdl/post_training` 组件
- 外部依赖, 权重路径, 数据路径要在 docstring 写清楚
- `core/` 内文件被 tests 或 `docs/md/` 引用时, 移动或改名必须同步更新引用

## 验证建议

- 玩具入口 (`core/pretrain/train_*.py`, `core/posttrain/train_GRPO.py`)
  修改后应可直接 `python` 运行闭环
- 大模型入口至少检查导入, 配置和主流程自洽
