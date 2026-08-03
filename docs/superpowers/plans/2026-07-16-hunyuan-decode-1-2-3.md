# HunyuanOCR Decode Perf Plan: 1 / 2 / 3

Date: 2026-07-16
Baseline: whole-block CUDA Graph mean ~0.102 ms (decode_min_int8_rows=16, dense fallback)

## Goals

1. **M=1 INT8 no-pad kernel** - keep W8A8 semantics without padding rows to 16/64
2. **GQA single-token rewrite** - stop padding query to 64 rows
3. **Projection fusion / fewer launches** - cut independent Linear launches inside one block

## P0 Research Facts (locked)

| Path | M=1 result | Note |
| --- | --- | --- |
| `torch._int_mm` | fails for M<=16 | needs M>16 |
| TileLang `T.gemm` INT8 | needs `block_m >= 16` | MMA warp floor on sm_89 |
| PTX sm_89 | works M=1 | often slower than dense GEMM for these shapes |
| dense `F.linear` (current default) | fastest M=1 | loses native INT8 compute |
| TileLang interactive JIT in REPL | `OSError: could not get source code` | kernels must live in real `.py` files |

## Acceptance

- Numeric: block output matches composed reference within existing rtol/atol
- Perf: real-shape CUDA Graph mean around 0.10 ms for default path
- Tests: decode + hunyuan_block pytest green
- Docs: `docs/md/usage/xqt-hunyuan-ocr.md` updated

## Execution Order

### P1 - M=1 INT8 GEMV (no pad) [DONE]

- Added `int8_linear_static_activation_m1_tilelang` (float32 products of int8 codes; no pad)
- Wired into `Int8MmaLinear` when `rows==1` and static fused path is selected
- Default remains `decode_min_int8_rows=16` (dense faster); set 0 to force INT8 no-pad

### P2 - GQA single-token kernel [DONE]

- `query_tile_rows=1` uses exact reference GQA (no query pad)
- Default `gqa_query_tile_rows=1`; legacy 64 still available

### P3 - Projection fusion [DONE]

- QKV and gate+up fused pack for M=1 when `decode_min_int8_rows=0` (INT8 path)
- Dense default path keeps sequential projections (faster for cuBLAS small-M)
- CUDA Graph capture-safe (no device sync inside capture)

### P4 - Verify [DONE]

- pytest decode + hunyuan_block: 19 passed
- real-shape graph:
  - default: ~0.107 ms (gqa=1, dense M=1)
  - int8_fusion (`decode_min_int8_rows=0`): ~0.243 ms

### P5 - Docs [DONE]

- Documented in `docs/md/usage/xqt-hunyuan-ocr.md`

## Non-goals

- Fix full-model W8A8 OCR quality (already diagnosed as per-tensor activation issue)
- Replace HF generate path
- Change training / quantizer defaults globally

## Resulting defaults

| Field | Default | Meaning |
| --- | --- | --- |
| `decode_min_int8_rows` | 16 | M=1 dense `bf16_fallback` (fast path) |
| `gqa_query_tile_rows` | 1 | exact single-token GQA |
| `int8_block_m` | 16 | multi-token / forced INT8 MMA |

Force full 1/2/3 INT8 path: `decode_min_int8_rows=0` (enables M=1 INT8 no-pad + QKV/gate-up fusion).
