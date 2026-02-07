# RWKV-8 (Eagle) 算子实现学习项目

本项目从零开始实现 RWKV-8 的核心算子，包含详细的数学推导、算法分析和性能优化。

## 项目结构

```
rwkv8/
├── docs/               # 数学推导和算法文档
├── reference/          # 参考实现（PyTorch）
├── cuda/              # CUDA 核心算子实现
├── benchmarks/        # 性能测试和对比
└── experiments/       # 学习实验和调试工具
```

## RWKV-8 核心创新

### 1. Multi-Head Latent Attention (MLA)
- 将 Key/Value 压缩到 latent space
- 大幅减少 KV cache 内存占用

### 2. 动态衰减机制改进
- 引入 gate 机制控制信息流
- 更好的长距离依赖建模

### 3. 数值稳定性优化
- 改进的 WKV 计算方式
- 更好的梯度流动

## 学习路径

1. **理论基础**: 阅读 `docs/` 中的数学推导
2. **参考实现**: 理解 `reference/` 中的 PyTorch 代码
3. **CUDA 实现**: 学习 `cuda/` 中的高性能算子
4. **性能调优**: 运行 `benchmarks/` 中的测试

## 构建和运行

```bash
# 构建 CUDA 算子
cd cuda && make

# 运行测试
python -m pytest tests/

# 运行基准测试
python benchmarks/compare_speed.py
```

## 参考资料

- [RWKV 论文](https://arxiv.org/abs/2305.13048)
- [RWKV-4 训练说明](https://github.com/BlinkDL/RWKV-LM)
- [State Space Models 综述](https://arxiv.org/abs/2312.00752)
