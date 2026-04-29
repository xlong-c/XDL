# TODO-P1-1: model/AGENTS.md 更新

> **优先级**: P1 | **预估工时**: 1h | **风险**: 极低

---

## 问题

`xdl/model/AGENTS.md` 与代码不一致：

| 遗漏 | 实际 |
|------|------|
| `vit_huge_patch14_224` | 代码存在 (`vit.py:255`) |
| `SimpleMLP` / `simple_mlp` | 代码存在 (`simple_mlp.py`) |
| 注册表计数 | 写的是示例条目，非完整的 28 个 |
| `FATT` / `segment/` 目录 | 存在但未说明 |
| `TwinFlow` 使用说明 | 无示例 |

---

## 修复

1. 更新模型表格 — 列出完整 28 个条目（ResNet 6 + VGG 9 + ViT 7 + TwinFlow 1 + MLP 2 + 组件 5）
2. 添加 TwinFlow YAML 配置示例
3. 说明 `segment/` 目录和 FATT 状态
4. 更新计数为准确数字

---

## 验证

```bash
python -c "
from xdl.model import MODEL_REGISTRY
names = set(MODEL_REGISTRY.list_available())
assert len(names) >= 28, f'Expected >=28, got {len(names)}'
# 检查关键条目
for n in ['vit_huge_patch14_224','SimpleMLP','TwinFlow']:
    assert n in names, f'Missing {n}'
print('All models accounted for')
"
```

---

## 完成标准

- [x] 28 个注册条目在文档中列出
- [x] TwinFlow 有使用示例
- [x] FATT 状态已说明
