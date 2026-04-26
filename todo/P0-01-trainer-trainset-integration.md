# P0-01: Trainer.fit() 与 TrainSetup 无法衔接

## 概述

YAML 配置流可以成功构建所有训练组件并返回 `TrainSetup` 对象，但 `Trainer` 无法消费这个对象——两条体系的接口完全不兼容。目前 `setup_from_yaml` 只有测试在调用，没有训练脚本能接入。

## 涉及文件

| 文件 | 关键位置 |
|------|----------|
| `xdl/trainer/trainer.py:239` | `Trainer.fit()` 签名 |
| `xdl/trainer/trainer.py:48` | `Trainer.__init__()` 签名 |
| `xdl/trainer/coreModel.py:153` | `CoreModel` 类定义 |
| `xdl/config/dataclass.py:14` | `TrainSetup` dataclass |
| `xdl/config/setup.py:249` | `TrainSetup` 组装 |
| `xdl/model/vgg.py:11` | VGG（nn.Module，非 CoreModel） |
| `xdl/model/resnet.py:139` | ResNet（nn.Module，非 CoreModel） |
| `xdl/model/vit.py:133` | ViT（nn.Module，非 CoreModel） |

## 具体差异

### Trainer.fit() 的要求

```python
# xdl/trainer/trainer.py:239
def fit(
    self,
    model: CoreModel,                          # 必须是 CoreModel 子类
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    val_check_interval: Union[int, float] = 1.0,
    check_val_every_n_epoch: int = 1,
    inference_data: Optional[Any] = None,
):
```

- `model` 必须是 `CoreModel` 子类，不是普通 `nn.Module`
- **不接受** `optimizer`、`loss_fn`、`scheduler`、`metrics` 参数
- 期望 model 内部通过 `configure_optimizers()` 返回 optimizer/scheduler
- 期望 model 的 `training_step()` 自行调用 `self.loss_fn`

### TrainSetup 提供的内容

```python
# xdl/config/dataclass.py:38-57
@dataclass
class TrainSetup:
    model: torch.nn.Module                    # 普通 nn.Module，不是 CoreModel
    train_loader: DataLoader
    optimizer: torch.optim.Optimizer          # 外部构建的优化器
    loss_fn: torch.nn.Module                  # 外部构建的损失函数
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    scheduler: Optional[Any] = None
    metrics: List[Any] = field(default_factory=list)
    full_config: Dict[str, Any] = field(default_factory=dict)
    device: str = "cuda"
    num_epochs: int = 100
    batch_size: int = 128
```

### 模型注册现实

所有通过 `register_model` 注册的模型都是普通 `nn.Module`，没有一个是 `CoreModel`：

- `VGG(nn.Module)` — `xdl/model/vgg.py:11`
- `ResNet(nn.Module)` — `xdl/model/resnet.py:139`
- `VisionTransformer(nn.Module)` — `xdl/model/vit.py:133`
- `SimpleMLP(nn.Module)` — `xdl/model/simple_mlp.py:9`
- `TwinFlow(nn.Module)` — `xdl/model/generate/twinflow.py:8`

唯一的 `CoreModel` 子类是 `train_VAE.py` 中的 `VAEModel`（脚本内部定义，未注册）。

### 当前实际的训练流程（方式 1）

```python
# train_VAE.py
class VAEModel(CoreModel):          # 手动写 CoreModel 子类
    def training_step(self, batch, batch_idx):
        ...
        loss = self.loss_fn(...)    # loss_fn 在 model 内部
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

trainer = Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)  # 只传 model + dataloader
```

## 影响

- YAML 配置流（方式 2）完全无法用于实际训练
- `xdl/config/` 整个模块形同虚设，只有测试用例在运行
- 新用户按文档写 YAML 配置后会发现在 `Trainer.fit()` 处无法继续

## 修复方向

两个方向，需要选一个：

**方案 A：扩展 Trainer，接受外部组件**
- `Trainer.fit()` 增加 `optimizer`、`loss_fn`、`scheduler`、`metrics` 参数
- model 可以是普通 `nn.Module`，不强求 `CoreModel`
- Training loop 内部用传入的 loss_fn 计算 loss，用传入的 optimizer 更新参数
- 优点：YAML 构建的外部组件直接可用；普通 `nn.Module` 模型零改造
- 缺点：Trainer 需要大改；与 `CoreModel` 的设计理念冲突

**方案 B：让 YAML 构建出的模型变成 CoreModel**
- 提供一个适配层，将外部 optimizer/loss_fn/scheduler/metrics 注入到一个包装 CoreModel 中
- 或者改造 `build_model` 使其返回 `CoreModel` 子类
- 优点：Trainer 不变
- 缺点：需要在构建阶段做额外包装；与注册的普通 nn.Module 不兼容

**方案 C：提供 train_from_yaml() 一键函数**
- 新增 `train_from_yaml(config_path)` 函数，内部同时创建 Trainer 和组件，处理所有衔接逻辑
- 用户只调一个函数，不直接接触 Trainer
- 优点：对用户最简单
- 缺点：只是把衔接问题藏起来，内部仍需解决 A 或 B

无论选哪个方向，都需要选择一个方案后统一推进。
