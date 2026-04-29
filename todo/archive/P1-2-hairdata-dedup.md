# TODO-P1-2: 数据集增强代码去重 (hairdata 3 文件)

> **优先级**: P1 | **预估工时**: 3h | **状态**: ✅ DONE

---

## 问题

`hairdata.py` / `hairdata3y.py` / `hairdata10hair.py` 共享 ~60% 代码：
- `pixel_transform` / `hair_transform` — 完全相同
- `self.norm` / `self.to_tensor` — 完全相同
- `refer_imgaug` / `imgaug` — 仅 crop 尺寸不同 (768 / 512 / None)
- `denormalize` — 完全相同（`__main__` 块）

## 修复

1. **新建** `xdl/dataset/hair_transforms.py` (97 行):
   - `create_pixel_transform()` / `create_hair_transform()` — 工厂函数
   - `HairAugMixin` — mixin 类, 提供 `_init_hair_aug(crop_size)`, `imgaug`, `refer_imgaug`

2. **修改** 3 个 hairdata 文件:
   - 继承 `HairAugMixin` + 调用 `_init_hair_aug(crop_size=...)` 替代重复的变换定义
   - 删除 `imgaug` / `refer_imgaug` 方法定义, 移除无用 import

## 效果

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| 总行数 | 741 (235+237+269) | 550 (97+143+145+165) |
| 重复行 | ~180 | 0 |
| 净减少 | — | -191 (-26%) |

## 验证

```python
python -c "
import ast
for f in ['hair_transforms.py', 'hairdata.py', 'hairdata3y.py', 'hairdata10hair.py']:
    with open(f'xdl/dataset/{f}') as fp:
        ast.parse(fp.read())
from xdl.dataset.hairdata import GridImageDataset as G1
from xdl.dataset.hairdata3y import GridImageDataset as G2
from xdl.dataset.hairdata10hair import Hair10HairDataset
assert issubclass(G1, HairAugMixin)
assert issubclass(G2, HairAugMixin)
assert issubclass(Hair10HairDataset, HairAugMixin)
print('OK: All 3 datasets use shared HairAugMixin')
"
```

## 完成标准

- [x] 新建 `hair_transforms.py` 共享模块
- [x] 三个类继承 `HairAugMixin`
- [x] crop_size 参数化 (768 / 512 / None)
- [x] 等价行为不变
