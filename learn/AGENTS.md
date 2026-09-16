# learn — 学习与实验目录

## 目录职责

- 存放算法学习代码、CUDA/Triton 实验、教程草稿与验证脚本
- 用于快速试验，不直接代表 XDL 主框架稳定接口

## 当前内容

- `attention/`、`flash_attention/`、`image_generative/`：算法学习实验
- `cuda/`、`nunchaku/`、`triton/`：底层实现与性能实验
- `bp_tutorial/`、`quant/`：教程与专题笔记
- `math/`: 数学教程与专题笔记, 改动前先遵循 `learn/math/AGENTS.md`

## 修改约束

- 允许更强实验性，但要把实验目标写清楚
- 不把 `learn/` 中的临时实现直接当作 `xdl/` 正式接口
- 需要进入主框架的能力，先整理边界再迁移
- CUDA/C++/Triton 代码可按实验需要组织，但文件命名和注释仍应可读
- 新增或重构面向人类阅读的 HTML 教程页时,遵循 `docs/md/architecture/html-style-policy.md`. 默认复用 `docs/html/assets/xdl-doc.css` 的主题 token 和公共组件;实验性页面可以有局部组件,但不要复制整套内联样式,公共版式或另写主题系统

## 注意事项

- `learn/` 下代码风格可以比主框架松，但不要失去可追溯性
