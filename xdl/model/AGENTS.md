# xdl/model - 模型架构模块

## 目录职责

- 提供可注册的模型类与工厂函数
- 为纯代码训练和 YAML 配置构建提供模型入口

## 当前内容

- `resnet.py`:ResNet 系列
- `vgg.py`:VGG 系列
- `vit.py`:Vision Transformer 系列
- `simple_mlp.py`:SimpleMLP 与默认工厂
- `generate/twinflow.py`:TwinFlow 生成模型
- `segment/fatt.py`:FATT 分割模型
- `lowlevel/`:底层超分模型

当前 `MODEL_REGISTRY` 覆盖 (实际条目以 `MODEL_REGISTRY.list_available()` 为准):

- ResNet:`ResNet`,`resnet18/34/50/101/152`
- VGG:`VGG`,`vgg11/11_bn/13/13_bn/16/16_bn/19/19_bn`
- ViT:`VisionTransformer`,`vit_tiny/small/base/large/huge`
- MLP:`SimpleMLP`,`simple_mlp`
- 生成:`TwinFlow`,`TBSMGenerator`,`IdentityRepresentor`,`ScatteringTracker`,`RepresentationScatteringField`,`DistributionMatcher`,`RepresentationScattering`
- 分割:`FATT`
- 底层 SR:`RGT`,`RGT_S`,`ATD`,`RRDBNet`,`OFTSR_UNet`,`OFTSR_SuperResModel`,`AutoEncoder_RRDBNet`,`ProbabilisticAutoEncoder_RRDBNet`,`RealPLKSR`
- 可选研究模型:`RealPLKSR_Ult`,`WFEN`,`DAT_2` (需 `spandrel`),`ESC`; 依赖缺失时跳过注册并告警

内部构件 `BasicBlock`,`Bottleneck`,`PatchEmbedding`,`MultiHeadAttention`,`TransformerBlock` 保持可从 `xdl.model` 导入, 但不注册为 MODEL, 避免稀释 registry 的模型语义.

## 修改约束

- 新模型先放到合适的子文件,再在 `xdl/model/__init__.py` 中集中注册
- 不在定义处使用装饰器式注册;本仓约定是集中注册
- 研究型模型用 `_register_optional_models()` 的可选注册路径, 不在模块顶层硬导入重依赖或版本敏感依赖
- 模型文件优先只放结构实现,不混入训练脚本逻辑
- 模型参数假设,输入尺寸假设和外部依赖要写清楚

## 使用方式

纯代码:

```python
from xdl.model import resnet50, SimpleMLP

cnn = resnet50(num_classes=100)
mlp = SimpleMLP(in_features=784, hidden_dims=[512, 256], out_features=10)
```

YAML:

```yaml
model:
  target: "registry:resnet50"
  params:
    num_classes: 100
```

框架内构建统一走 registry. 独立推理脚本 (`infer/`) 需要可选依赖时, 可直接从 `xdl.model.lowlevel.*` 导入对应研究模型.

## 验证建议

- 改注册后检查 `MODEL_REGISTRY.list_available()` 是否包含新条目
- 至少补一个最小 `forward` 形状验证
- 影响配置构建时同步检查 `setup_from_yaml()` 主路径
