# YAML 配置启动体系审查 — 问题清单

审查范围：`xdl/config/setup.py`、`builder.py`、`resolver.py`、`schema.py`、`dataclass.py`，以及它们与 `xdl/trainer/trainer.py` 的对接关系。

所有问题均附带：问题位置（文件路径 + 行号）、现象、影响和修复建议。

| 优先级 | 编号 | 问题 | 涉及文件 |
|--------|------|------|----------|
| **P0** | #1 | Trainer.fit() 与 TrainSetup 无法衔接 | setup.py, trainer.py, dataclass.py |
| **P0** | #2 | logging / checkpoint / accelerate 配置被静默丢弃 | setup.py, schema.py, dataclass.py |
| **P1** | #3 | config_version 声明但不校验 | schema.py, resolver.py |
| **P2** | #4 | build_transform 重复调用 _normalize_transform_shorthand | builder.py |
| **P2** | #5 | _resolve_dataset_value 中存在死代码 | setup.py |
| **P2** | #6 | 三层收集函数完全重复 | setup.py |
| **P2** | #7 | setup_from_yaml 内 resolve 流程绕路 | setup.py, resolver.py |
| **P2** | #8 | TrainSetup.train_loader 类型标注不准确 | dataclass.py |
| **P2** | #9 | batch_size fallback 失败时静默回退到 1 | setup.py |
| **P2** | #10 | Trainer.setup_logger() **kwargs 吞掉未知参数 | trainer.py |

详细描述见各编号文件。
