# Tools Directory

`tools/` 存放 XDL 框架外围工具。新增脚本时优先放入现有分类，并在脚本顶部写清输入、输出和副作用。

## 分类

- `agent/`：Agent 配置、Skill、MCP 辅助工具。
- `data_processing/`：通用文件、CSV、parquet 等数据处理。
- `dataset/`：数据集下载、整理、复制、重命名、清洗。
- `experiments/`：一次性或研究期工具，保留需写清上下文。
- `image/`：图像批处理、检测、去重、查看。
- `ml/`：模型侧辅助脚本，例如预编码、推理检查、ComfyUI 辅助。
- `setup/`：环境准备与缓存检查。
- `system/`：系统级操作脚本，通常有较强副作用。
- `test/`：硬件或设备测试。
- `video/`：视频处理。

## 约束

- Python 脚本不使用 `argparse`，优先使用 YAML 或代码内显式配置。
- 删除、覆盖、移动、批量重命名必须默认可审计，必要时提供 dry-run。
- 不提交明文 token、运行缓存、个人 notebook 输出或明显过期入口。
