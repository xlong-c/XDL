# XQT 导出前前置融合

## 负责内容

- 定义 `xqt` 在 ONNX 导出前的前置融合能力.
- 说明它在 `PyTorch -> ONNX -> QDQ -> backend compile` 链路中的位置.
- 约束首期支持的融合模式,配置入口,校验要求和非目标.

## 不负责内容

- 不负责删除 ONNX 里的 `QuantizeLinear` / `DequantizeLinear`.
- 不替代 TensorRT,OpenVINO,ONNX Runtime 等后端的图优化和 kernel 融合.
- 不承诺覆盖所有模型族,动态控制流模型或任意自定义模块结构.
- 不把本文中的内部实现细节提升为 Stable API.

## 背景

XQT 当前主场景之一是 `PyTorch baseline -> ONNX export -> ONNX Runtime QDQ INT8 -> TensorRT/OpenVINO 友好产物 -> benchmark/manifest`,见 [XQT.md](XQT.md). 这条链路里,量化阶段会显式在 ONNX 图中插入 QDQ 节点,但部署后端通常需要先看到一个足够规整的 float 图,才能稳定识别低精度子图和后续 kernel 融合机会.

导出前前置融合的目标不是替代后端,而是在 PyTorch `eval` 模型上先折叠一小类数学上安全,工程价值明确的模块组合,减少训练态残留结构,让后续 ONNX 导出,`onnxsim`,QDQ 插点和后端编译更稳定.

## 在导出链中的位置

推荐顺序:

```text
PyTorch float model
    -> eval()
    -> pre-export fusion
    -> torch.onnx.export
    -> optional onnxsim
    -> optional ONNX Runtime QDQ
    -> TensorRT / OpenVINO / ONNX Runtime compile
```

职责划分:

- 前置融合: 清理 PyTorch float 图中的经典安全模式.
- ONNX exporter: 保持输入输出签名,把 PyTorch 图变成 ONNX.
- `onnxsim`: 清理导出后残留的无效 reshape/transpose/identity 等通用冗余.
- QDQ 量化: 把量化语义显式写入 ONNX.
- 部署后端: 吸收和融合可识别的 QDQ 及低精度子图.

## 适用场景

### 1. CNN / ResNet / 检测骨干网络

这是首期的核心场景. 这些模型常见 `Conv + BatchNorm (+ ReLU)` 模式,前置融合收益明确:

- ONNX 图更简洁.
- QDQ 边界更稳定.
- 后端更容易命中 `Conv` 类 INT8 kernel.

### 2. 静态 PTQ / QDQ 前处理

在 `onnxruntime_qdq` 路径中,如果量化前 float 图还保留大量可提前折叠的 BN 或激活边界,量化后图会更复杂,也更难排查误差来源. 前置融合的作用是先把 float 图规整好.

### 3. 需要稳定数值回归和导出排障的场景

前置融合把一部分可预期变化提前收敛到 PyTorch 侧,后续做 PyTorch vs ONNX vs backend diff 时更容易定位问题来源.

## 非目标场景

以下场景首期不建议依赖本功能:

- ViT / LLM / DiT 这类缺少大量 `Conv + BN` 模式的模型.
- 大量函数式写法,动态控制流或模块边界不清晰的模型.
- 把前置融合当作“删 QDQ 节点”手段的需求.
- 需要跨残差,跨 attention 或自定义多分支模式的大范围图重写.

## 首期支持的融合模式

首期只支持 PyTorch 官方已有稳定支持的经典模式:

- `conv + bn`
- `conv + bn + relu`
- `conv + relu`
- `linear + relu`
- `bn + relu`

实现上优先复用 PyTorch `torch.ao.quantization.fuse_modules` 和 `torch.ao.quantization.quantize_fx.fuse_fx`,不自己手写数学折叠逻辑.

当前实现位置:

- `xqt/export/fusion.py`: `apply_pre_export_fusion()` 和结果 metadata.
- `xqt/export/onnx_exporter.py`: 在 ONNX 导出前应用融合并写入导出 metadata.
- `xqt/pipeline/passes.py`: `export` pass 和 `onnxruntime_qdq` 自动导出路径接收并透传该配置.
- `xqt/core/config.py`: 对 `export.targets[*].params.pre_export_fusion` 和 `compression.quant.policy.pre_export_fusion` 做配置校验.

## 模式发现策略

首期支持两类模式发现方式:

### 1. 显式模块列表

由配置直接给出 `modules_to_fuse`,例如:

```yaml
export:
  targets:
    - format: onnx
      output_path: artifacts/xqt/model.onnx
      params:
        pre_export_fusion:
          enabled: true
          mode: eager
          modules_to_fuse:
            - ["layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu"]
            - ["layer1.0.conv2", "layer1.0.bn2"]
```

适用于模块层级清晰,需要精确控制融合范围的模型.

### 2. FX 自动融合

对模块结构规整,但手写模块路径成本高的模型,可使用 `mode: fx`,交给 `fuse_fx()` 在 `eval` 图上自动识别支持模式.

```yaml
export:
  targets:
    - format: onnx
      output_path: artifacts/xqt/model.onnx
      params:
        pre_export_fusion:
          enabled: true
          mode: fx
```

## 配置边界

前置融合属于 export 前处理,因此配置挂在 `export.targets[i].params.pre_export_fusion`.

首期支持字段:

- `enabled`: 是否开启,默认 `false`.
- `mode`: `eager` 或 `fx`,默认 `eager`.
- `modules_to_fuse`: 仅 `eager` 模式使用,为二维字符串列表.
- `inplace`: 是否原地修改输入模型,默认 `false`.

约束:

- `mode=eager` 时,`modules_to_fuse` 不能为空.
- `mode=fx` 时忽略 `modules_to_fuse`.
- 该配置只影响当前 export target,不自动传播到其他 target.
- ONNX QDQ 自动导出路径复用同一套策略,但配置来源为 `compression.quant.policy.pre_export_fusion`.

## 与其他阶段的职责划分

### 与 `onnxsim`

- 前置融合处理 PyTorch 模块组合.
- `onnxsim` 处理 ONNX 图层面的通用冗余.
- 两者不互相替代.

### 与 ONNX QDQ

- 前置融合发生在 float 图.
- QDQ 插点仍由量化 pass 决定.
- 前置融合不负责减少部署后端必须看到的量化语义边界.

### 与后端融合

- 前置融合负责“把图导得更规整”.
- 后端负责“把可识别子图编译成更优实现”.

## 数值和工程约束

1. 只允许在 `model.eval()` 后执行.
2. 默认不原地修改调用方传入模型.
3. 导出侧应该记录是否应用了前置融合,使用的模式以及模块列表. 当前 ONNX export metadata 和 ONNX QDQ quant metadata 都会保留 `pre_export_fusion`.
4. 若融合失败,默认抛出清晰错误,不要静默降级.
5. 前置融合后的模型仍应通过现有导出校验和 runtime diff.

## 验证要求

实现完成后至少应验证:

1. `eager` 模式能对显式 `Conv + BN + ReLU` 模型成功融合.
2. `fx` 模式能在简单顺序模型上生成可导出的 fused graph.
3. 不合法配置会抛出明确错误.
4. ONNX 导出结果 metadata 能反映 fusion 是否启用.
5. pipeline 中 `export` 和 `onnxruntime_qdq` 自动导出路径都能接收该配置.

## API 边界

本能力属于 `xqt.export` 内部实现,当前按 Internal 对待.

- 可以通过 `xqt.export` 模块复用内部 helper.
- 不对 `xdl/` 主框架暴露稳定兼容承诺.
- 行为变化应同步更新本文和 [XQT.md](XQT.md) 中与导出链有关的说明.
