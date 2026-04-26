# P2-08: TrainSetup.train_loader 类型标注不准确

## 涉及文件

`xdl/config/dataclass.py:38-39`

## 问题代码

```python
@dataclass
class TrainSetup:
    model: torch.nn.Module
    train_loader: DataLoader                    # 标注为 DataLoader，非 Optional
    optimizer: torch.optim.Optimizer            # 标注为 Optimizer，非 Optional
    loss_fn: torch.nn.Module                    # 标注为 nn.Module，非 Optional
    val_loader: Optional[DataLoader] = None
    ...
```

但在 `setup_from_yaml` 的构建过程（`setup.py:251`）：

```python
train_loader=built_dataloaders.get("train"),
optimizer=optimizer,
loss_fn=loss_fn,
```

- 如果 YAML 没有 `train_dataloader` 块，`built_dataloaders.get("train")` 返回 `None`
- 如果 YAML 没有 `optimization.optimizer`，已经在 `setup.py:224` 抛出 `ConfigValidationError`
- 如果 YAML 没有 `loss`，已经在 `setup.py:232` `build_loss([])` 中抛出 `ConfigValidationError`

所以 optimizer 和 loss_fn 在运行时不会为 None（有显式校验），但 `train_loader` 可能为 None（无显式校验）。

## 影响

- 静态类型检查器（mypy / pyright）不会标记此问题
- 运行时如果 `train_loader=None`，后续 `Trainer.fit(model, None, ...)` 会崩溃在 `len(train_dataloader)`（`trainer.py:282`），报错不够明确
- dataclass 自身的类型不匹配（`None` 赋值给 `DataLoader` 类型字段）在 Python 中不抛异常，导致问题延迟到更远的调用处才暴露

## 修复建议

在 `setup_from_yaml` 返回前增加显式校验：

```python
if built_dataloaders.get("train") is None:
    raise ConfigValidationError("train_dataloader config is required")
```

同时将类型标注改为更诚实的 `Optional[DataLoader]`，或者确认 `train_loader` 永远是必需的（不是 Optional），在注释中明确。
