# config — 运行配置目录

## 目录职责

- 存放训练/推理的 YAML 配置
- 存放 DeepSpeed 与分布式 JSON 配置
- 作为 `setup_from_yaml()` 的输入样例

## 当前内容

- `vgg_cifar100.yaml`：基础分类训练示例
- `unified_logger_example.yaml`：日志与回调配置示例
- `deepspeed_*.json`：不同 ZeRO 策略模板

## 修改约束

- 优先复用 `xdl/config/` 的 schema、resolver、builder 机制
- 新增 YAML 样例应兼容 `OmegaConf` 解析、插值和 resolver 规则；复杂默认值复用优先使用 `${...}` 插值，而不是要求训练脚本手写拼接
- 不写机器私有绝对路径、token、密钥
- 新增配置尽量保持最小可运行，字段名与代码参数保持一致
- 训练入口仍以 YAML 为主，不在这里引入命令行参数解析

## 验证建议

- 涉及配置构建时优先跑 `tests/config/`
- 复杂配置至少用一次 `setup_from_yaml()` 做构建验证
