# 性能报告约定

性能结论必须拆分为:

- 首次 compile time.
- steady-state step time.
- host input pipeline time.
- metric transfer time.
- checkpoint time.

`xdl_jax.performance.benchmark_callable()` 会单独报告 compile 和
steady-state timing,并写入当前 backend,设备列表,warmup 和 repeat.
不要把首次编译混入稳态 step,也不要只用端到端 wall time 宣称 kernel 性能.

当前已经有 CPU/GPU 单设备和 2-device CPU correctness 证据. 输入管线,
metric transfer 和 checkpoint 独立计时仍是后续性能工程项,不是当前版本的
吞吐承诺.
