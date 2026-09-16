# 架构文档入口

本文汇总 `XDL` 的架构文档. 架构层负责定义边界, 模块关系, 生命周期, 稳定性和禁止项.

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
- [xdl-improvement-goals.md](xdl-improvement-goals.md): `XDL` 改进任务的唯一状态源, 含详细目标, 依赖和验收标准.
- [xdl-jax.md](xdl-jax.md): `xdl-jax` 独立 JAX 训练架构, 边界与限制.
- [xdl-jax-refactor-and-alignment.md](xdl-jax-refactor-and-alignment.md): `xdl-jax` 抽象抽离与接口对齐优化方案.

## 当前兼容事实源

- [../README.md](../README.md): 旧 `XDL` 总文档事实源, 仍保留兼容入口角色.

## 推荐阅读顺序

### XDL

1. [xdl.md](xdl.md)
2. [dataset-policy.md](dataset-policy.md)
3. [module-boundaries.md](module-boundaries.md)
4. [api-boundary.md](api-boundary.md)
5. [html-style-policy.md](html-style-policy.md)
6. [roadmap.md](roadmap.md)
7. [xdl-improvement-goals.md](xdl-improvement-goals.md)
