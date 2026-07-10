# 说明文档入口

本文汇总 `XDL` 与 `XQT` 的说明文档. 说明层负责解释概念, 术语, 认知模型和常见误区.

## 负责什么

- 解释框架核心概念.
- 解释为什么这样组织.
- 给人类和 agent 建立共同认知.

## 不负责什么

- 不单独定义新契约.
- 不承载完整操作步骤.
- 不替代架构层的边界定义.

## 当前文档

- [xdl-concepts.md](xdl-concepts.md): `XDL` 的概念地图和理解路径.
- [xqt-concepts.md](xqt-concepts.md): `XQT` 的概念地图和边界理解.
- [backends/index.md](backends/index.md): XQT 导出 / runtime backend 和 operator optimization engine 分文档入口.
- [flux2-klein-nvfp4-backends.md](flux2-klein-nvfp4-backends.md): FLUX.2 klein NVFP4 的 `cutedsl`, `cutile`, `tilelang` 三类 XQT engine 推理接入.
- [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md): 手写 CUDA / CUTLASS / CuTe DSL 算子的调优方法论, 覆盖 GEMM, Conv, Linear, Attention, Norm, 访存, 通信, 低精度, 以及 NVIDIA / AMD 代际 `MMA` / `Matrix Core` 总表.
- [inference-backends-primer.md](inference-backends-primer.md): 推理后端和模型部署格式的基础介绍, 不绑定 `XQT` 实现.
- [w4-int8-mma-retarget.md](w4-int8-mma-retarget.md): W4 存储 + INT8 MMA 计算转义的设计, 实现, 策略对照与基准验收.
- [mma-weight-prepack.md](mma-weight-prepack.md): 离线 MMA 权重预排板.
- [dataset-structure.md](dataset-structure.md): dataset 模块的概念说明和阅读路径.
- [html-style.md](html-style.md): HTML 阅读页的分层说明.
- `HTML`: 阅读版入口见 [../../html/index.html](../../html/index.html) 和 [../../html/xqt.html](../../html/xqt.html).

## 推荐阅读顺序

### XDL

1. [xdl-concepts.md](xdl-concepts.md)
2. [dataset-structure.md](dataset-structure.md)
3. [html-style.md](html-style.md)
4. [../../html/index.html](../../html/index.html)

### XQT

1. [xqt-concepts.md](xqt-concepts.md)
2. [backends/index.md](backends/index.md)
3. [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md)
4. [w4-int8-mma-retarget.md](w4-int8-mma-retarget.md)
5. [mma-weight-prepack.md](mma-weight-prepack.md)
6. [inference-backends-primer.md](inference-backends-primer.md)
7. [../XQT_SUMMARY.md](../XQT_SUMMARY.md)
8. [../../html/xqt.html](../../html/xqt.html)
