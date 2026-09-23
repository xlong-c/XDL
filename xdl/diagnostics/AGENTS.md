# xdl/diagnostics - 训练性能诊断模块

## 目录职责

- 提供 XDL 训练的低开销 GPU 利用率 / 功耗 / 显存诊断能力
- 把每步阶段耗时, 同步点, 显存锯齿, GPU 采样聚合为根因 finding 与改进建议
- 纯逻辑包 (本包) 与生命周期适配器 (`xdl/callbacks/diagnostics_callback.py`) 分离

## 当前内容

- `records.py`:数据契约 (PhaseSample/PhaseAggregate, GpuSample/GpuSummary,
  MemorySummary, SyncCounters/SyncSummary, Suggestion, Finding, DiagnosticReport)
- `phase_timer.py`:`PhaseTimer`, 按 mark 序列聚合每步阶段耗时
- `gpu_sampler.py`:`GpuSampler`, 后台 NVML 线程采样利用率/功耗/显存/温度
- `sync_tracker.py`:`ReservedMemoryTracker`, 检测 `empty_cache()` 造成的显存锯齿
- `analyzer.py`:`AnalyzerConfig` / `AnalysisContext` / `DiagnosticAnalyzer`, 规则引擎
- `deep_dive.py`:`DeepDiveRecorder`, 按需触发 `torch.profiler` 深挖
- `report.py`:`build_report` / `render_text` / `write_report`

## 核心约束

- 纯逻辑包不 import torch; 保证无卡环境可单测
- 可选依赖 (pynvml/psutil/torch.profiler) 必须优雅降级, 导入不失败
- 常驻开销目标 < 2%:热循环只用守卫布尔 + `time.perf_counter`
- 诊断异常绝不打断训练; 报告失败降级为 warning
- 只在主进程写报告

## 接入方式

框架内 YAML 入口 (schema v1) 开启即可自动挂载:

```yaml
diagnostics:
  enabled: true
  report_dir: logs/diagnostics
  deep_dive: true
```

`Trainer.from_setup()` 会按配置追加 `DiagnosticsCallback`。手动入口直接
`Trainer(callbacks=[..., DiagnosticsCallback(report_dir="logs/diagnostics")])`。

Trainer 侧接线: `_diag_enabled` / `_phase_timer` / `_diag_counters` 三个属性由
回调在 `on_train_start` 注入; 关闭时热循环只多一次布尔判断。

## 验证建议

- `python -m pytest tests/diagnostics -q` (纯 CPU, 无卡可跑)
- 真实入口冒烟: 挂 `DiagnosticsCallback` 跑 `train/core/posttrain/train_GRPO.py`
- 改动规则阈值后同步更新 `tests/diagnostics/test_analyzer.py`