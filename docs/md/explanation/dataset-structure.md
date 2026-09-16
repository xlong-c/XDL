# Dataset 结构说明

本文作为 `XDL` dataset 模块的说明层落点. 它解释 dataset 相关概念和阅读路径, 不替代详细模板规划或 HTML 结构页.

## 这是什么

这是给人类和 agent 建立 dataset 认知的说明页. 它负责回答:

- `xdl/dataset` 里主要有哪些角色
- manifest / folder / sidecar 等数据组织方式分别适合什么场景
- 什么时候该看概念页, 什么时候该看事实源或 HTML 结构图

## 为什么需要

dataset 相关内容天然跨越三层:

- 架构层需要定义模块边界
- 说明层需要解释概念和选择方式
- 使用层需要给出配置和扩展动作

如果全部堆进一个总文档, 会让"概念理解"和"操作步骤"互相干扰.

## 核心概念

- `record`: 以 manifest 为中心的通用样本组织方式
- `folder`: 纯目录扫描或按目录分类的图片组织方式
- `sidecar`: 用 basename 对齐附加文本, mask 或其他伴随文件的组织方式
- `transform`: 单样本层的数据变换
- `collate`: batch 层的数据拼接逻辑

## 该看哪里

- 看详细事实边界和模板规划: [../architecture/dataset-policy.md](../architecture/dataset-policy.md)
- 看整体框架定位: [xdl-concepts.md](xdl-concepts.md)

## 常见误区

- 不要把 dataset 模板选择和训练 workflow 混成一页
- 不要只看 YAML 字段名猜返回结构
- 不要绕过 registry 新起一套并行接入链路
