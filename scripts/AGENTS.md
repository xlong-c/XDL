# scripts — 仓库维护脚本目录

## 目录职责

- 存放安装、打包、检查、离线处理等仓库级脚本
- 支撑开发环境准备与日常维护任务
- 不存放个人实验入口、模型辅助脚本、系统盘制作脚本或 Agent 私人工具

## 当前内容

- `install.sh`：安装脚本
- `build_wheel.py`：自动构建 XDL wheel，默认输出到 `dist/`
- `check_dependency_config.py`：依赖配置检查
- `offline_bundle.py`：离线资源处理

## 常用命令

```bash
python scripts/build_wheel.py
XDL_BUILD_CLEAN=1 python scripts/build_wheel.py
```

`build_wheel.py` 不使用命令行参数解析库，行为通过环境变量控制：`XDL_BUILD_OUT_DIR`、`XDL_BUILD_CLEAN`、`XDL_BUILD_CLEAN_DIST`、`XDL_BUILD_BACKEND`、`XDL_BUILD_CHECK`。

## 修改约束

- Python 脚本继续遵守仓库约束：不要引入 `argparse`
- 脚本优先可重复执行，副作用要明确
- 涉及路径、权限、删除动作时先写清楚假设
- 与训练/推理强耦合的逻辑应留在对应目录，不要全部堆进 `scripts/`
- 与 XDL 仓库维护无关但仍有用的脚本优先放到 `tools/` 对应子目录

## 验证建议

- 改动 shell 脚本后至少检查关键路径与错误分支
- 改动 Python 脚本后补最小可运行说明
