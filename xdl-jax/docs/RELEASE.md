# 发布门槛

`0.1.0` 发布候选至少需要:

- `pyright xdl_jax` 通过.
- `ruff check` 通过.
- CPU full test 通过.
- wheel 可以构建并在临时目录导入.
- CPU synthetic example 和 YAML example 可运行.
- checkpoint exact/weights-only 测试通过.
- GPU 测试只在具备对应 CUDA plugin 的机器上运行.
- CHANGELOG 和已知限制与实际证据一致.

当前明确限制:

- 只有单卡 GPU 已有真实运行证据.
- 2-device 仅有 CPU correctness 证据.
- 多平台,多 GPU,多主机和 TPU 不作无证据承诺.
- NumPy source 不提供精确 iterator resume.
- XQT 交接只提供模型侧 artifact helper,尚未把它作为跨项目 CI gate.

