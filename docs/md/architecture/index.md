# 架构文档入口

本文汇总 `XDL` 与 `XQT` 的架构文档. 架构层负责定义边界, 模块关系, 生命周期, 稳定性和禁止项.

## 负责什么

- 定义系统是什么, 不是什么.
- 定义模块边界和依赖方向.
- 定义生命周期, 扩展点和兼容层级.

## 不负责什么

- 不提供完整安装手册.
- 不承载长 FAQ 或页面样式展示.
- 不重复具体工作流命令.

## 当前文档

- [xdl.md](xdl.md): `XDL` 定位, 核心分层, 生命周期, 模块边界.
- [dataset-policy.md](dataset-policy.md): `XDL` dataset 模板选择与新增规则.
- [module-boundaries.md](module-boundaries.md): `XDL` 子模块职责, 依赖方向与高影响结构改动.
- [api-boundary.md](api-boundary.md): `XDL` Stable / Provisional / Internal 边界与废弃策略.
- [html-style-policy.md](html-style-policy.md): `XDL` 自有 HTML 阅读页长期规范.
- [roadmap.md](roadmap.md): `XDL` 当前仍有效的优化方向.
- [xqt.md](xqt.md): `XQT` 定位, 模型侧边界, Stage 约定, profiling 约定.
- [xqt-kernel-wrapper-nn-boundary.md](xqt-kernel-wrapper-nn-boundary.md): `kernel` / `wrapper/materialize` / `xqt.nn` 三层边界.
- [xqt-operator-block-optimization.md](xqt-operator-block-optimization.md): `single_kernel` / `block_kernel` 分层, block 准入和手写融合路径.
- [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md): `engine` / `quant` 词表, 三轴, 禁止项与配置语义.
- [xqt-infer-handoff.md](xqt-infer-handoff.md): quant→infer 交接面 (`model` + `compute_config`).
- [xqt-design-debt.md](xqt-design-debt.md): `XQT` 设计债台账, 只累计问题, 后面统一做方案.
- [xqt-realignment-guide.md](xqt-realignment-guide.md): `XQT` 现状诊断 1-8 的落地状态和长期矫正准则.

## 当前兼容事实源

- [../README.md](../README.md): 旧 `XDL` 总文档事实源, 仍保留兼容入口角色.
- [../XQT.md](../XQT.md): 旧 `XQT` 长期事实源, 仍保留兼容入口角色.
- [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md): `XQT` 包内工程契约.

## 推荐阅读顺序

### XDL

1. [xdl.md](xdl.md)
2. [dataset-policy.md](dataset-policy.md)
3. [module-boundaries.md](module-boundaries.md)
4. [api-boundary.md](api-boundary.md)
5. [html-style-policy.md](html-style-policy.md)
6. [roadmap.md](roadmap.md)

### XQT

1. [xqt.md](xqt.md)
2. [xqt-kernel-wrapper-nn-boundary.md](xqt-kernel-wrapper-nn-boundary.md)
3. [xqt-operator-block-optimization.md](xqt-operator-block-optimization.md)
4. [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md)
5. [xqt-infer-handoff.md](xqt-infer-handoff.md)
6. [xqt-design-debt.md](xqt-design-debt.md)
7. [xqt-realignment-guide.md](xqt-realignment-guide.md)
8. [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md)
