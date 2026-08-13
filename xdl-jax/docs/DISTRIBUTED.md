# 分布式边界

当前正式候选能力分为两层:

- `SingleDeviceStrategy`: CPU/GPU/TPU 设备选择接口,本机已验证 CPU 和单卡 GPU.
- `DataParallelStrategy`: 单主机一维 data mesh,已用 2-device CPU 做 correctness
  验证,不等同于多主机生产支持.

第一阶段的 data parallel contract 是:

```text
batch -> PartitionSpec("data", ...)
model/optimizer state -> replicated
gradient/metrics -> data-axis pmean
```

多 GPU,TPU,多主机,Tensor/Pipeline/Expert Parallel 和跨拓扑 checkpoint
不在当前正式验收范围内. 无目标硬件时必须保持未验收状态.

