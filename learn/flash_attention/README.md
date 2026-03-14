# FlashAttention Evolution

This directory contains conceptual demonstrations and architectural notes tracing the evolution of FlashAttention from v1 to v4, focusing on memory hierarchy utilization, hardware-kernel synchronization, and quantization.

## Evolution Overview

- **[v1: Tiling & Kernel Fusion](flash_attention_v1.py)** - Introduced tiling to reduce HBM read/write cycles.
- **[v2: Scheduling Optimization](flash_attention_v2.py)** - Focused on kernel scheduling, work partitioning, and reducing non-MatMul operations.
- **[v3: Hardware Fusion (Hopper)](flash_attention_v3.py)** - Leveraged asynchronous TMA units, warp specialization (Producer-Consumer), and interleaved GEMM-Softmax pipelines.
- **[v4: Blackwell Adaptability](flash_attention_v4.py)** - Further optimized for B200's TMEM architecture, FP8 quantization, and incoherent processing via Hadamard transforms.
