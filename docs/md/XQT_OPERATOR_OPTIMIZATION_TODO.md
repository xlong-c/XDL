# XQT 算子优化与 Megakernel TODO

本文档负责定义 `xqt` 中算子优化,megakernel 和定制 kernel 的建设顺序. 它是 TODO 和实施边界文档,不是已实现 API 承诺.

当前仓库事实:

- `xqt` 已有 `pre_export_fusion`,用于导出前图规整,见 `xqt/export/fusion.py`.
- `xqt` 已有 `benchmark/latency.py`,可复用为 steady-state latency 基线.
- `pyproject.toml` 的 `optimization` optional extra 已包含 `torchao`,`triton`,`tilelang`.
- `learn/tilelang/flashatt.py` 已有 TileLang FlashAttention forward 学习示例.
- `learn/rwkv/rwkv8/rwkv8_tilelang.py` 已有 TileLang ROSA suffix-match 教学 kernel.
- `learn/deepseek/mHC/mhc_tilelang.py` 已有 TileLang mHC 学习实现.
- `xqt/operator_opt` 已有 pass,capability,preflight,manifest,`torch_compile` 执行路径,Triton/TileLang/CuTile/CUTLASS/custom CUDA 的 metadata/fallback/guard 骨架.
- `xqt/operator_opt/backends/` 是后端 adapter 分包,`xqt/operator_opt/kernels/` 是算子 reference 和 guarded entry 分包. 旧的 `xqt/operator_opt/*_backend.py` 和 `*_kernels.py` 已移除,不再兼容旧路径.
- Triton 目前有真实 `@triton.jit` 入口和 CPU fallback 测试. 默认沙箱进程仍看不到 CUDA runtime,但已在非沙箱 GPU 进程完成 smoke 验证: RTX 4070 Ti SUPER,driver 591.86,CUDA driver API 13.1,PyTorch 2.10.0+cu130,`sm_89`.
- TileLang 已从 `learn/tilelang/flashatt.py` 抽取 attention tile/online-softmax/causal-mask 设计 metadata,但生产入口仍是 reference guarded path,不是已实证的 TileLang JIT kernel.
- CuTile 和 CUTLASS Python 目前是可审计 scaffold: package capability,reference,CPU fallback,artifact metadata 和 preflight,尚未接入真实 DSL kernel.

## 1. 边界

算子优化不等同于量化,剪枝或导出.

- `quant` 负责 dtype,scale,observer,calibration 和量化 artifact.
- `prune` 负责结构化/非结构化稀疏和模块重写.
- `pre_export_fusion` 负责导出前图规整,让 ONNX/TensorRT/OpenVINO 更容易识别后端 fusion.
- `operator_opt` 负责 PyTorch runtime 或定制 kernel runtime 下的算子替换,编译,megakernel 和 kernel-level benchmark.

推荐 pipeline 位置:

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
- 算子优化应在模型结构和 dtype 稳定后执行.
- 导出型 runtime 仍应优先交给目标后端做 kernel fusion,不要把 PyTorch custom kernel 硬塞到 ONNX 链路.

## 2. 后端分层

算子优化按成本和风险分层推进.

| 层级 | 后端 | 定位 | 优先级 | 备注 |
| --- | --- | --- | --- | --- |
| 自动编译 | `torch_compile` | 通过 `torch.compile`/Inductor/CUDA Graphs 消化通用 fusion 和 launch overhead | P0 | 首个可落地 backend |
| 部署后端 | `deployment_backend` | TensorRT,OpenVINO,ONNX Runtime 自己的 graph/kernel fusion | P0 | XQT 只做配置,preflight 和 benchmark |
| Triton | `triton` | 小到中等规模定制 kernel,pointwise/reduction/epilogue 优先 | P1 | 适合快速验证 fused SwiGLU,RMSNorm,RoPE |
| TileLang | `tilelang` | tile/block 结构明显的高性能 kernel,attention,GEMM epilogue,dequant GEMM,MLA 类优先 | P1 | 插在 Triton 与 C++/CUDA 之间,作为实验后端 |
| CuTile | `cutile` | nvvc CuTile Python DSL 路径,服务 CUDA tile DSL kernel 原型和后续生产化 | P1/P2 | 当前只做 metadata/reference/fallback scaffold |
| CUTLASS Python | `cutlass` | CUTLASS Python/CuTe DSL 路径,面向 GEMM epilogue,grouped GEMM 和特定精度矩阵路径 | P1/P2 | 当前只做 metadata/reference/fallback scaffold |
| C++/CUDA | `custom_cuda` | Triton/TileLang 表达困难或需要更底层控制的生产 kernel | P2 | optional extension,不污染基础安装 |

TileLang 定位:

- TileLang 不是量化 backend,也不是通用导出 backend.
- TileLang 是 `operator_opt` 下的定制 kernel backend.
- TileLang 优先服务 tiled dataflow 明确的场景: attention,MLA/paged attention,dequant GEMM,GEMM epilogue,block sparse,2:4 sparse,复杂 shared memory pipeline.
- 简单 pointwise chain 首选 Triton 或 `torch.compile`,不要为了使用 TileLang 而增加维护成本.
- TileLang,CuTile,CUTLASS Python kernel 默认视为 PyTorch runtime only,除非后续明确实现 TensorRT plugin 或其他部署桥接.

## 3. 适用模块

| 模型族 | 首批候选 | 推荐路径 |
| --- | --- | --- |
| CNN/ResNet | Conv-BN-Act,Conv-Bias-Act,postprocess | 先 `pre_export_fusion` + ONNX/TensorRT/OpenVINO,少量 PyTorch runtime 才走 `torch_compile` |
| ViT/Transformer | LayerNorm/RMSNorm + residual,QKV reshape,RoPE,MLP epilogue,attention | `torch_compile` -> Triton/TileLang kernel -> benchmark |
| LLM decoder | RMSNorm,RoPE,KV cache append,paged attention,SwiGLU,logits sampling | 组件级 compile,attention 优先复用成熟库,TileLang 做 tiled attention/MLA 实验 |
| Diffusion UNet/DiT | GroupNorm + SiLU,attention block,scheduler step,VAE postprocess | 固定 shape 时优先 CUDA Graphs/compile,TileLang 用于 attention/GEMM 类重热点 |
| 异构模型 | vision encoder,projector,decoder 分别优化 | 组件级 policy,禁止强行跨 runtime/device 合并成一个 megakernel |

不适合第一期做 megakernel 的对象:

- shape 高度动态且会频繁触发 recompile 的路径.
- 已经由 TensorRT/ONNX Runtime/OpenVINO 高质量融合的导出图.
- 大 GEMM 全量自研替代 cuBLASLt/TensorRT/Inductor.
- 跨组件,跨 device,跨 dtype 的 giant kernel.

## 4. 配置草案

待新增 schema:

```yaml
operator_optimization:
  enabled: true
  stage: after_compression
  default_backend: torch_compile
  targets:
    - name: decoder_compile
      target: decoder
      backend: torch_compile
      mode: reduce-overhead
      fullgraph: false
      dynamic: false
      options:
        max_autotune: true
        epilogue_fusion: true
        triton.cudagraphs: true

    - name: decoder_mlp_triton
      target: decoder.layers.*.mlp
      backend: triton
      patterns:
        - swiglu
        - bias_gelu
      fallback: eager
      min_speedup: 1.10
      validate:
        atol: 1e-4
        rtol: 1e-4

    - name: decoder_attention_tilelang
      target: decoder.layers.*.self_attn
      backend: tilelang
      patterns:
        - attention
        - dequant_gemm
      tilelang:
        target: cuda
        target_arch: null
        threads: 128
        num_stages: 2
        cache_dir: null
        pass_configs:
          TL_ENABLE_FAST_MATH: true
      fallback: eager
      min_speedup: 1.15
      validate:
        atol: 1e-3
        rtol: 1e-3
```

配置规则:

- 不新增命令行参数解析库.
- YAML 通过 OmegaConf structured config 接入.
- `target` 支持组件名,模块路径或通配 pattern,但第一期只承诺显式模块路径.
- `backend: tilelang` 必须有 CUDA/HIP 等可用 device,否则 preflight warning 或 fail.
- `backend: cutile` 和 `backend: cutlass` 第一阶段只承诺 `target_arch`,cache,pass config,registry 和 fallback metadata,不承诺真实 DSL 执行.
- `fallback` 默认 `eager`,任何数值或性能验收失败都不得静默替换.
- `min_speedup` 必须大于 `1.0`,否则该优化没有进入 pipeline 的意义.

## 5. 代码改动清单

### P0: 文档和 schema

- [x] 新增 `docs/md/XQT_OPERATOR_OPTIMIZATION_TODO.md`.
- [x] 在 `docs/md/README.md` 增加索引和改动前必读项.
- [x] 在 `docs/md/XQT.md` 挂接 `operator_opt` 方向.
- [x] 在 `xqt/core/schema.py` 增加 `OperatorOptimizationConfig`.
- [x] 在 `xqt/core/schema.py` 增加 `OperatorOptimizationTargetConfig`.
- [x] 在 `xqt/core/schema.py` 增加 `OperatorOptimizationValidationConfig`.
- [x] 在 `xqt/core/schema.py` 增加 `TileLangKernelConfig`.
- [x] 在 `xqt/core/schema.py` 增加 `CuTileKernelConfig`.
- [x] 在 `xqt/core/schema.py` 增加 `CutlassKernelConfig`.
- [x] 在 `xqt/core/config.py` 校验 backend 枚举: `torch_compile`,`deployment_backend`,`triton`,`tilelang`,`cutile`,`cutlass`,`custom_cuda`.
- [x] 在 `xqt/core/config.py` 校验 target name 唯一.
- [x] 在 `xqt/core/config.py` 校验 `min_speedup > 1.0`.
- [x] 在 `xqt/core/config.py` 校验 TileLang pass config key 只允许已知字符串,未知 key 先拒绝.
- [x] 在 `xqt/core/config.py` 校验 CuTile pass config key 只允许已知字符串,未知 key 先拒绝.
- [x] 在 `xqt/core/config.py` 校验 CUTLASS Python tile/cluster shape 和 pass config key.

### P0: torch.compile pass

- [x] 新增 `xqt/operator_opt/__init__.py`.
- [x] 新增 `xqt/operator_opt/types.py`.
- [x] 新增 `xqt/operator_opt/capability.py`.
- [x] 新增 `xqt/operator_opt/compile_backend.py`.
- [x] 新增 `xqt/operator_opt/executor.py`.
- [x] 新增 `OperatorOptimizationPass`.
- [x] 在 `xqt/pipeline/runner.py` 接入 `operator_optimization.enabled`.
- [x] 在 `xqt/pipeline/passes.py` 接入 pass 执行顺序.
- [x] `torch_compile` backend 支持整模型 compile.
- [x] `torch_compile` backend 支持按组件 compile.
- [x] 记录 compile time,steady-state latency,backend mode,fullgraph,dynamic,options.
- [x] compile 失败时按 `fallback` 回退,并在 metrics 记录 skip reason.

### P0: preflight 和 manifest

- [x] 在 `xqt/pipeline/preflight.py` 增加 operator optimization capability 检查.
- [x] 检查 PyTorch 版本是否支持 `torch.compile`.
- [x] 检查 CUDA 是否可用,并记录 device capability.
- [x] 检查 `triton` 是否可 import.
- [x] 检查 `tilelang` 是否可 import.
- [x] recipe 启用 `cutile` 时检查 `cutile` 是否可 import.
- [x] recipe 启用 `cutlass` 时检查 `cutlass` 是否可 import.
- [x] TileLang 可用时记录版本,target,target_arch 和 cache 配置.
- [x] TileLang 不可用但 recipe 启用 `backend: tilelang` 时给出明确 fail/warning.
- [x] CuTile/CUTLASS 不可用但 recipe 启用对应 backend 时给出明确 warning.
- [x] manifest 新增 `operator_optimization` section.
- [x] manifest 记录每个 target 的 backend,runtime,applied,fallback,latency,numeric_diff.

### P1: profiler 和候选发现

- [x] 新增 `xqt/benchmark/profiler.py`.
- [x] 用 `torch.profiler` 收集 operator latency,kernel count,CUDA time,self CUDA time,memory.
- [x] 新增 `xqt/operator_opt/patterns.py`.
- [x] 支持 FX graph candidate scan.
- [x] 支持 `torch.export` graph candidate scan.
- [x] 输出候选 pattern: `bias_gelu`,`swiglu`,`rmsnorm_residual`,`rope`,`attention`,`dequant_gemm`,`qdq_epilogue`.
- [x] 候选报告包含 shape,dtype,device,estimated_memory_io,estimated_kernel_count,recommended_backend.
- [x] 候选报告不得自动替换模块,只给建议.

### P1: 分包结构和多后端 scaffold

- [x] 新增 `xqt/operator_opt/backends/` 后端 adapter 分包.
- [x] 新增 `xqt/operator_opt/kernels/` 算子 reference 和 guarded entry 分包.
- [x] 移除 `xqt/operator_opt/*_backend.py` 和 `*_kernels.py` 兼容 re-export,统一使用分包路径.
- [x] 新增 `xqt/operator_opt/backends/cutile.py` 和 `xqt/operator_opt/kernels/cutile/`.
- [x] 新增 `xqt/operator_opt/backends/cutlass.py` 和 `xqt/operator_opt/kernels/cutlass/`.
- [x] CuTile 示例算子 `bias_silu` 提供 reference,CPU fallback,CUDA guard,registry 和 artifact metadata.
- [x] CUTLASS Python 示例算子 `gemm_epilogue` 提供 reference,CPU fallback,CUDA guard,registry 和 artifact metadata.
- [x] executor 对 Triton/TileLang/CuTile/CUTLASS/custom CUDA 统一记录 metadata 并明确 skip,实际模型替换仍只由 `torch_compile` 执行.

### P1: Triton backend

- [x] 新增 `xqt/operator_opt/backends/triton.py`.
- [x] 新增 `xqt/operator_opt/kernels/triton/`.
- [x] 实现 `fused_bias_gelu` reference + Triton kernel.
- [x] 实现 `fused_swiglu` reference + Triton kernel.
- [x] 实现 `fused_rmsnorm_residual` reference + Triton kernel.
- [x] 实现 `fused_rope` reference + Triton kernel.
- [x] 每个 kernel 支持 CUDA-only skip.
- [x] 每个 kernel 提供 eager fallback.
- [x] 每个 kernel 记录 block size,num warps,num stages,autotune key.
- [x] 每个 kernel 有数值对齐测试和 latency smoke.

### P1: TileLang backend

- [x] 新增 `xqt/operator_opt/backends/tilelang.py`.
- [x] 新增 `xqt/operator_opt/kernels/tilelang/`.
- [x] 从 `learn/tilelang/flashatt.py` 抽取可复用 attention kernel 设计,不要直接把学习脚本变成生产接口.
- [x] 从 `learn/rwkv/rwkv8/rwkv8_tilelang.py` 抽取 TileLang wrapper 经验: CUDA 输入检查,CPU reference,明确错误提示.
- [x] 定义 TileLang kernel registry: pattern -> kernel factory -> capability.
- [x] 定义 TileLang pass config 映射,避免 recipe 直接传任意 Python 对象.
- [x] 支持 `target: cuda` 第一阶段; HIP/CPU/Metal 只记录为未来 capability,不承诺执行.
- [x] 支持 `target_arch` 显式配置,用于无可见 GPU 但可编译的环境.
- [x] 支持 `cache_dir`,默认不写到仓库目录.
- [x] TileLang 编译产物路径和版本写入 artifact metadata.
- [x] TileLang kernel 默认只在 CUDA tensor 上执行,CPU 自动 fallback.
- [x] 第一个 TileLang 生产候选优先选择 `fused_attention_forward` 或 `dequant_gemm_epilogue`,不要从小 pointwise 开始.
- [x] 对比基线必须包含 PyTorch SDPA 或 eager reference.
- [x] TileLang 数值阈值按 dtype 分开配置: fp16/bf16 默认比 fp32 宽.
- [x] TileLang benchmark 需要分开记录 compile latency 和 execution latency.

### P2: 量化联动

- [x] `operator_opt` 读取 `context.metrics["quant"]` 的 runtime 和 backend.
- [x] `onnxruntime_qdq` 产物默认不进入 PyTorch custom kernel 替换.
- [x] `torchao` PyTorch runtime 可以进入 `torch_compile` 和局部 Triton/TileLang 优化.
- [x] planned backend `gptq`,`awq`,`bitsandbytes` 只输出 capability,不执行 kernel 替换.
- [x] 新增 `dequant_gemm_epilogue` 候选分析.
- [x] 新增 `weight_only_matmul_epilogue` 候选分析.
- [x] 新增 FP8 scale/cast/matmul epilogue 候选分析.
- [x] 对量化后 kernel 的数值验收使用量化阈值,不复用 FP32 严格阈值.

### P2: 导出联动

- [x] 明确 `operator_opt` 产物的 exportability.
- [x] `torch_compile` optimized model 默认标记为 PyTorch runtime artifact.
- [x] Triton/TileLang/custom CUDA 默认标记为 non-exportable.
- [x] CuTile/CUTLASS Python 默认标记为 non-exportable.
- [x] 导出到 ONNX 时应使用原始模型或导出友好模型,不能依赖 PyTorch custom kernel.
- [x] 如果后续支持 TensorRT plugin,必须作为单独 backend capability,不能隐式复用 TileLang kernel.
- [x] `pre_export_fusion` 保持在 export path 内,不要迁移到 `operator_opt`.

### P3: C++/CUDA custom op

- [x] 新增 `xqt/operator_opt/cuda_extension.py`.
- [x] 使用 optional extension,基础安装不编译 nvcc 代码.
- [x] 使用 `torch.library` 注册 custom op.
- [x] 提供 fake/meta implementation 供 compile/export 分析.
- [x] 提供 `opcheck` 或等价测试.
- [x] 支持 wheel 构建开关和环境 preflight.
- [ ] 仅在 Triton/TileLang 无法覆盖或性能差距明确时使用.

## 6. 首批 recipe

- [x] `xqt/recipes/operator_compile_smoke_cpu.yaml`: CPU 上验证 schema,pass,fallback,manifest.
- [x] `xqt/recipes/operator_compile_cuda.yaml`: CUDA 上验证 `torch_compile` latency. 2026-06-20 在非沙箱 RTX 4070 Ti SUPER 上实测: recipe 可生成真实 CUDA latency manifest,最近一次 run 为 compile time 305.27 ms,mean latency 0.0713 ms -> 0.1464 ms,speedup 0.487x,未达到 `min_speedup` 因而按预期不替换,device `cuda:0`,dtype `torch.float32`;artifact 写入 `/tmp/xqt-operator-compile-cuda/operator_optimization.json`.
- [x] `xqt/recipes/operator_vit_compile.yaml`: ViT/Transformer 组件级 compile.
- [x] `xqt/recipes/operator_llm_mlp_triton.yaml`: Triton MLP epilogue smoke.
- [x] `xqt/recipes/operator_attention_tilelang.yaml`: TileLang attention smoke,CUDA-only.
- [x] `xqt/recipes/operator_quant_torchao_compile.yaml`: torchao 量化后接 `torch_compile`.
- [x] `xqt/recipes/operator_qdq_export_guard.yaml`: QDQ artifact 不走 PyTorch custom kernel,只验证导出链不被破坏.

## 7. 测试清单

- [x] `tests/xqt/test_operator_opt_config.py`: schema 加载和非法配置.
- [x] `tests/xqt/test_operator_opt_preflight.py`: backend capability,TileLang import,CUDA skip.
- [x] `tests/xqt/test_operator_opt_pass.py`: pass 顺序,metrics,fallback.
- [x] `tests/xqt/test_operator_opt_compile.py`: `torch_compile` smoke.
- [x] `tests/xqt/test_operator_opt_patterns.py`: FX/export candidate scan.
- [x] `tests/xqt/test_operator_opt_triton.py`: Triton kernel 数值对齐和 CUDA skip.
- [x] `tests/xqt/test_operator_opt_tilelang.py`: TileLang kernel 数值对齐,CUDA skip,compile-only smoke.
- [x] `tests/xqt/test_operator_opt_backends.py`: 分包 import,CuTile/CUTLASS reference,fallback,metadata 和 pass 报告.
- [x] `tests/xqt/test_runner.py`: operator optimization recipe 进入 runner.
- [x] `tests/xqt/test_manifest.py`: manifest 包含 operator optimization artifact 和 metrics.

测试约束:

- 无 CUDA 环境不能失败,应 skip CUDA/Triton/TileLang 执行测试.
- TileLang compile-only 测试必须把 cache 放到 `/tmp` 或测试临时目录.
- 不把本地 GPU 型号写死到测试.
- 性能测试只做 smoke,真实 speedup 验收放手动 benchmark 或 GPU CI.

## 8. 验收标准

每个 target 必须输出:

- `target_name`
- `module_path`
- `backend`
- `runtime`
- `applied`
- `fallback`
- `skip_reason`
- `compile_time_ms`
- `latency_before`
- `latency_after`
- `speedup`
- `numeric_diff`
- `device`
- `dtype`
- `shape_signature`
- `exportable`
- `artifact_paths`

最低验收:

- 数值对齐通过.
- `speedup >= min_speedup` 才允许默认启用替换.
- fallback 路径输出和原模型一致.
- manifest 能解释为什么替换或为什么跳过.
- 同一 recipe 重跑时不会污染仓库目录.

## 9. TileLang 进入 XQT 的原则

TileLang 要进 XQT,必须满足:

- 有 PyTorch reference.
- 有最小 CUDA correctness test.
- 有无 CUDA skip.
- 有 compile-only 检查.
- 有明确 shape,dtype,device capability.
- 有 cache 和 artifact 管理.
- 有性能对比,至少对比 PyTorch eager/SDPA 或 `torch.compile`.
- 有 fallback.
- 有文档说明不可导出的 runtime 边界.

TileLang 不做:

- 不作为 XQT 默认必需依赖.
- 不替代 `torch.compile`.
- 不替代 TensorRT/OpenVINO/ONNX Runtime 后端 fusion.
- 不承诺一套 kernel 覆盖所有 GPU 架构.
- 不在第一期承诺训练 backward kernel.

## 10. 执行顺序建议

第一轮只做低风险闭环:

1. schema + preflight + pass skeleton.
2. `torch_compile` backend.
3. manifest 和 benchmark 输出.
4. CPU smoke + CUDA skip 测试.

第二轮做候选发现:

1. profiler.
2. FX/export pattern scan.
3. 输出建议,不替换.

第三轮接定制 kernel:

1. Triton `fused_swiglu`.
2. Triton `fused_rmsnorm_residual`.
3. TileLang attention smoke.
4. TileLang dequant GEMM epilogue 预研.

第四轮再考虑生产化:

1. 组件级异构策略.
2. 量化后 kernel 联动.
3. TensorRT plugin 或 custom CUDA extension 单独立项.
