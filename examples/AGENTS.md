# examples — 示例脚本目录

## 目录职责

- 存放可运行的训练、微调或最小示例脚本
- 展示 XDL 与外部库的接入方式

## 当前内容

- `flux2_klein_9b_lora_finetune.py`：LoRA 微调示例
- `mlp_simulation.py`：轻量示例
- 其他脚本覆盖不同实验入口

## 修改约束

- 示例应尽量最小但完整，优先体现真实调用路径
- 遵守仓库规则：不要引入 `argparse`，优先显式配置或 YAML
- 外部依赖、权重路径、数据路径要写清楚
- 示例不承担底层抽象职责，共享逻辑应下沉到 `xdl/` 或 `tools/`

## 验证建议

- 修改后至少检查导入、配置和主流程是否自洽
- 涉及训练生命周期时对照 `train_VAE.py`、`train_TwinFlow.py` 和 `xdl/trainer/`
