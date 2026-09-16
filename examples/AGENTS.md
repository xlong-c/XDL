# examples - 示例脚本目录

## 目录职责

- 存放可运行的示例脚本, 展示 XDL 与外部库的接入方式
- 正式训练入口不在这里: 预训练在 `train/pretrain/`, 后训练 (LoRA 微调,
  RL 等) 在 `train/posttrain/`, 见 [../train/AGENTS.md](../train/AGENTS.md)

## 当前内容

- `mlp_simulation.py`: 轻量示例
- `sefi_dit/`: SeFi 官方 finetune runbook (不走 XDL 路径)
- 其他脚本覆盖推理演示与实验入口

## 修改约束

- 示例应尽量最小但完整,优先体现真实调用路径
- 遵守仓库规则:不要引入命令行参数解析库,优先显式配置或 YAML
- 外部依赖,权重路径,数据路径要写清楚
- 示例不承担底层抽象职责,共享逻辑应下沉到 `xdl/` 或 `tools/`
- 新的训练任务入口放 `train/pretrain/` 或 `train/posttrain/`, 不放这里

## 验证建议

- 修改后至少检查导入,配置和主流程是否自洽
- 涉及训练生命周期时对照 `train/pretrain/train_VAE.py` 和 `xdl/trainer/`
