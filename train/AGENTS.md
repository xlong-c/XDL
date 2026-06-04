# train — 训练入口目录

## 目录职责

- 存放训练入口脚本与配套 YAML
- 承载外部模型或实验项目接入 XDL Trainer 的主流程

## 当前内容

- `train_sd35m_apex_xdl.py`：训练入口
- `sd35m_apex_lora.yaml`：配套训练配置

## 修改约束

- 编写新训练入口前，先看根目录 `train_VAE.py`、`train_TwinFlow.py` 和 `xdl/trainer/`
- 遵守 XDL 手动优化与 `CoreModel.setup()` 生命周期约束
- 参数管理优先 YAML 或代码内显式配置，不引入命令行参数解析库；复杂 YAML 加载、合并、插值优先使用 `OmegaConf` 或 `xdl.config` 主链路
- 复杂保存、采样、可视化逻辑优先通过 callback 集成

## 验证建议

- 至少检查配置加载、模型 setup、优化器构建、训练步主路径是否连通
- 外部 `third_party/` 依赖路径先搜索真实位置，再决定接入方式
