# xdl/loss — 损失函数子模块

## 目录职责

- 定义预训练通用可注册损失函数与损失工厂
- 为训练脚本和 YAML 配置提供统一 loss 构建入口

## 当前内容

- `focal_loss.py`
- `huber_loss.py`
- `contrastive_loss.py`
- `dice_loss.py`
- 分类/重建/生成/分割/序列等损失见 `__init__.py`

蒸馏, 步数蒸馏与偏好优化损失 (DPO/STPO/GRPO) 属后训练组件, 已收拢到
`xdl/post_training/`, 不在本子模块.

## 修改约束

- 新增 loss 后必须在 `__init__.py` 中集中注册
- 后训练向 loss 放 `xdl/post_training/` 并在其 `__init__.py` 注册
- 接口保持可调用语义，输入输出张量维度假设要清楚
- 数值稳定性、reduction 语义、dtype/device 兼容性要优先考虑
- 工厂函数与类的注册名保持一致和可预测

## 验证建议

- 修改后优先跑 `tests/loss/`
- 至少覆盖正常路径和显著边界输入
