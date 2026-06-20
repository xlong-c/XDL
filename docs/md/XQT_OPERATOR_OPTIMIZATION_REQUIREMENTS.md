# XQT 算子优化需求边界

本文档定义 `xqt/operator_opt` 的需求边界,适用场景,后端选择,常见算子和验收口径. 它负责回答什么时候该做 kernel 优化,应该怎么接入,以及如何判断优化是否有效. 任务状态和实现清单见 [XQT_OPERATOR_OPTIMIZATION_TODO.md](XQT_OPERATOR_OPTIMIZATION_TODO.md).

当前实现状态:

- `torch_compile` 是当前唯一会在 executor 中真实替换模型组件的执行后端.
- `triton` 已有 `bias_gelu`,`swiglu`,`rmsnorm_residual`,`rope` 的 reference,真实 `@triton.jit` 入口,CPU fallback 和 CUDA skip 测试. 默认沙箱进程看不到 CUDA runtime,非沙箱 GPU 进程已在 RTX 4070 Ti SUPER,driver 591.86,CUDA driver API 13.1,PyTorch 2.10.0+cu130,`sm_89` 上跑通 Triton/TileLang/torch.compile CUDA smoke.
- `tilelang` 已有 `attention`,`dequant_gemm_epilogue` 的 reference,registry,artifact metadata,CUDA guard 和 fallback. `attention` 已从 `learn/tilelang/flashatt.py` 抽取 block tiling,online softmax,causal mask 和 SDPA baseline 设计 metadata,但还不是已实证的 TileLang JIT 生产 kernel.
- `cutile` 已作为 nvvc CuTile Python DSL 后端预留,当前有 `bias_silu` reference/fallback/metadata scaffold,未接真实 DSL kernel.
- `cutlass` 已作为 CUTLASS Python/CuTe DSL 后端预留,当前有 `gemm_epilogue`,`grouped_gemm` registry 和 `gemm_epilogue` reference/fallback/metadata scaffold,未接真实 DSL kernel.
- `custom_cuda` 当前只提供 optional extension scaffold,`torch.library` placeholder op,FakeTensor/meta 和 opcheck,不默认编译 nvcc.

## 1. 目标

`operator_opt` 的目标是把模型结构和 dtype 稳定之后的 PyTorch runtime 热点转成更少 launch,更少 memory traffic 或更适合目标硬件的执行路径.

必须提供:

- 可配置 target,backend,patterns,fallback,min_speedup 和数值阈值.
- 可导出的 manifest,说明每个 target 为什么替换或为什么跳过.
- PyTorch reference 或 eager fallback.
- 数值对齐和延迟基准.
- CUDA/device/dtype/shape capability 说明.
- cache/artifact 路径记录,且默认不写入仓库目录.

不负责:

- 不负责量化策略,dtype policy,observer 或 calibration;这些属于 `xqt.quant`.
- 不负责结构化剪枝,head/channel/block rewrite;这些属于 `xqt.prune`.
- 不负责导出前 ONNX/TensorRT/OpenVINO 图规整;这些属于 `xqt.export` 和 `pre_export_fusion`.
- 不承诺 PyTorch custom kernel 可导出到 ONNX. Triton/TileLang/CuTile/CUTLASS/custom CUDA 默认 `exportable=false`.

## 2. Pipeline 位置

推荐顺序:

```text
load model
    -> optional distill/prune/rewrite
    -> optional quant
    -> optional operator_opt
    -> optional export
    -> eval/benchmark/manifest
```

原因:

- 剪枝会改变 channel,head,hidden shape.
- 量化会改变 dtype,module wrapper 和 backend runtime.
- 算子优化要在 shape,dtype,runtime 稳定后做.
- 导出型 runtime 应优先让 TensorRT,OpenVINO,ONNX Runtime 自己做 graph/kernel fusion.

## 3. 后端选择

| 后端 | 首选场景 | 不适合场景 | 当前状态 |
| --- | --- | --- | --- |
| `torch_compile` | 通用 PyTorch module,固定 shape,launch overhead 明显,组件级优化 | graph break 很多,动态 Python 控制流复杂,后续必须导出 ONNX 的模型对象 | 可执行替换 |
| `deployment_backend` | TensorRT,OpenVINO,ONNX Runtime 的导出后 fusion | PyTorch runtime custom kernel | metadata only |
| `triton` | pointwise,reduction,normalization,RoPE,MLP epilogue,小到中等 fused kernel | 复杂 tiled attention 或大 GEMM 全量替代 | 有 JIT entry,已完成手动 GPU smoke |
| `tilelang` | attention,MLA,paged attention,dequant GEMM,GEMM epilogue,block sparse,2:4 sparse | 简单 pointwise chain,无明确 tile/block dataflow 的路径 | reference guarded scaffold |
| `cutile` | nvvc CuTile Python DSL 原型,CUDA tile DSL kernel,后续 Hopper/Blackwell 路径预研 | 没有 CuTile 环境或只是普通 pointwise 小融合 | reference guarded scaffold |
| `cutlass` | GEMM epilogue,grouped GEMM,FP8/FP4/NVFP4/MXFP 这类矩阵路径,MoE/adapter 多小 GEMM | 已由 cuBLASLt/TensorRT/Inductor 高质量覆盖的大 GEMM | reference guarded scaffold |
| `custom_cuda` | Triton/TileLang/CuTile/CUTLASS 表达困难,或性能证据明确要求更底层控制 | 没有性能证据,只是为了手写 CUDA | optional scaffold |

## 4. 常见算子

| pattern | 典型位置 | 推荐后端 | 输入形态 | 有效信号 |
| --- | --- | --- | --- | --- |
| `bias_gelu` | Transformer/ViT MLP epilogue | `triton` | `x[..., hidden] + bias[hidden]` | launch 数下降,小 batch 延迟下降 |
| `bias_silu` | Diffusion/MLP epilogue | `triton` 或 `cutile` | `x[..., hidden] + bias[hidden]` | pointwise chain memory traffic 下降 |
| `swiglu` | LLM decoder MLP | `triton` | `silu(gate) * up` | MLP epilogue launch 数下降 |
| `rmsnorm_residual` | Decoder block residual + norm | `triton` | `x,residual,weight` 共享 hidden dim | reduction + pointwise 融合有效 |
| `rope` | Attention 前 Q/K 旋转 | `triton` | even hidden dim,cos/sin 半维 | 小 kernel launch 减少 |
| `attention` | MHA/GQA/MLA/paged attention | `tilelang` 或成熟 attention 库 | `B,H,S,D` 或 block/paged cache | 对 SDPA/FlashAttention 有实测优势才启用 |
| `dequant_gemm_epilogue` | weight-only/int8/int4 Linear | `tilelang`,`cutlass` | `x,qweight,scale,bias` | dequant + matmul + epilogue 合并减少带宽 |
| `gemm_epilogue` | Linear-heavy path | `cutlass` | `x,weight,bias,activation` | CUTLASS 调度/epilogue 优于 eager/compile |
| `grouped_gemm` | MoE,多 adapter,多小 GEMM | `cutlass` | 多组 GEMM descriptor | kernel launch 合并和 occupancy 改善 |
| `kv_cache_append` | LLM decode | `custom_cuda` 或 DSL backend | cache layout 强相关 | profiling 证明 cache 更新是热点 |
| `sampling` | logits topk/topp/temperature | `triton` 或 `custom_cuda` | batch 小,词表大 | decode step 内 launch/latency 下降 |

## 5. 使用方式

最小配置:

```yaml
operator_optimization:
  enabled: true
  stage: after_compression
  targets:
    - name: model_compile
      target: null
      backend: torch_compile
      mode: reduce-overhead
      fallback: eager
      min_speedup: 1.01
```

Triton MLP epilogue smoke:

```yaml
operator_optimization:
  enabled: true
  targets:
    - name: mlp_triton
      target: mlp
      backend: triton
      patterns: [swiglu]
      fallback: eager
      min_speedup: 1.10
      validate:
        atol: 1e-4
        rtol: 1e-4
```

TileLang attention metadata/fallback path:

```yaml
operator_optimization:
  enabled: true
  targets:
    - name: attention_tilelang
      target: attention_block
      backend: tilelang
      patterns: [attention]
      fallback: eager
      min_speedup: 1.15
      tilelang:
        target: cuda
        target_arch: sm_89
        cache_dir: /tmp/xqt-tilelang-cache
        threads: 128
        num_stages: 2
        pass_configs:
          TL_ENABLE_FAST_MATH: true
```

CuTile scaffold:

```yaml
operator_optimization:
  enabled: true
  targets:
    - name: cutile_bias_silu
      target: mlp
      backend: cutile
      patterns: [bias_silu]
      fallback: eager
      min_speedup: 1.10
      cutile:
        target: cuda
        target_arch: sm_89
        cache_dir: /tmp/xqt-cutile-cache
        pass_configs:
          CUTILE_ENABLE_FAST_MATH: true
```

CUTLASS Python scaffold:

```yaml
operator_optimization:
  enabled: true
  targets:
    - name: cutlass_gemm_epilogue
      target: mlp
      backend: cutlass
      patterns: [gemm_epilogue]
      fallback: eager
      min_speedup: 1.10
      cutlass:
        target_arch: sm_89
        cache_dir: /tmp/xqt-cutlass-cache
        tile_shape: [128, 128, 64]
        pass_configs:
          CUTLASS_ENABLE_EPILOGUE_FUSION: true
```

## 6. 判断优化是否有效

一个优化只有同时满足下面条件,才能默认应用:

- 数值对齐: `numeric_diff.allclose=true`,并记录 `max_abs` 和 `mean_abs`.
- 延迟收益: `speedup >= min_speedup`.
- 稳定性: warmup 后 p50/mean 不因 compile 或 cache 抖动被误判.
- 适用性: shape,dtype,device 与 kernel capability 匹配.
- 可回退: 失败时明确 fallback 到 eager/reference,不能静默替换.
- 可解释: manifest 中能看到 backend,pattern,target,skip_reason,artifact_paths,latency 和 compile metadata.

建议用三层验收:

1. 单算子 correctness: reference vs backend output.
2. 组件级 latency: target module 前后 benchmark.
3. 任务级 regression: accuracy/perplexity/image metric 与 baseline 比较.

不要只看单次 benchmark. CUDA kernel 要记录 compile latency 和 steady-state latency;TileLang/CuTile/CUTLASS 这类 DSL 后端还要记录 cache 命中状态和 target arch.

## 7. GPU 验证要求

默认单测不能强依赖 GPU. GPU 验证可以作为手动 benchmark 或 GPU CI:

```bash
pytest tests/xqt/test_operator_opt_triton.py -q
pytest tests/xqt/test_operator_opt_tilelang.py -q
```

如果目标机器是 RTX 4070 Ti Super,推荐 recipe 中显式写 `target_arch: sm_89`. 如果当前进程 `torch.cuda.is_available()` 为 false,即使本机有 GPU,executor 也只能记录 metadata 并 skip CUDA backend. 这种情况通常是容器/沙箱/NVML/设备挂载限制,不是 recipe 语义错误.

当前手动验证记录:

- 2026-06-20,非沙箱 GPU 进程: RTX 4070 Ti SUPER,driver 591.86,CUDA driver API 13.1,PyTorch 2.10.0+cu130,`sm_89`.
- `pytest tests/xqt/test_operator_opt_triton.py tests/xqt/test_operator_opt_tilelang.py tests/xqt/test_operator_opt_compile.py -q`: 16 passed.
- `xqt/recipes/operator_compile_cuda.yaml`: recipe 可生成真实 CUDA latency manifest,最近一次 run 为 compile time 305.27 ms,mean latency 0.0713 ms -> 0.1464 ms,speedup 0.487x,未达到 `min_speedup` 因而按预期不替换,device `cuda:0`,dtype `torch.float32`,artifact `/tmp/xqt-operator-compile-cuda/operator_optimization.json`.

CUDA 13.1 环境下优先验证:

- PyTorch wheel 的 CUDA runtime 版本与驱动兼容.
- Triton 是否支持当前 Python/PyTorch/CUDA 组合.
- TileLang/CuTile/CUTLASS Python 的 package 版本和 target arch 是否支持 `sm_89`.
- cache_dir 是否在 `/tmp` 或 artifact 目录,不要写仓库源码目录.

## 8. 进入生产的准入线

Triton/TileLang/CuTile/CUTLASS/custom CUDA 从 scaffold 进入生产前,必须补齐:

- 真实 CUDA correctness 测试.
- 至少一个固定 shape 的 latency 对比.
- 对应后端 package/version/target_arch 的 preflight metadata.
- dtype 和 shape 限制文档.
- fallback 和 skip 行为测试.
- artifact/cache 记录.
- 任务级或组件级回归结果.

`custom_cuda` 额外要求:

- 只有 Triton/TileLang/CuTile/CUTLASS 无法覆盖,或性能证据明确显示必须使用更底层 CUDA 时才进入.
- 需要 optional build switch,基础安装不编译 nvcc.
- 需要 `torch.library` registration,fake/meta implementation 和 opcheck.
- 需要明确 wheel 构建策略和 ABI/CUDA 版本边界.
