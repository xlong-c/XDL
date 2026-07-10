"""W4A4 -> INT8 MMA acceptance via TileLang true INT8 MMA. No QAT, no training."""

from __future__ import annotations

import copy, statistics, time
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from xqt.quant.quantizers.fp4_weight_only import FP4WeightOnlyLinear, quantize_with_fp4_weight_only
from xqt.quant.quantizers.w4_storage_int8_mma import W4StorageInt8MmaLinear, quantize_with_w4_storage_int8_mma

@dataclass(frozen=True)
class BenchConfig:
    depth: int = 10
    width: int = 10240
    num_classes: int = 16
    batch_size: int = 64
    num_batches: int = 8
    seed: int = 11
    group_size: int = 128
    warmup: int = 15
    repeats: int = 60
    timing_trials: int = 5
    max_accuracy_drop_pp: float = 5.0
    min_speedup_b_vs_a: float = 1.15
    block_m: int = 64
    block_n: int = 64
    block_k: int = 64
    threads: int = 128
    num_stages: int = 2
    engine: str = "tilelang"

class DeepMLP(nn.Module):
    def __init__(self, *, depth: int, width: int, num_classes: int) -> None:
        super().__init__()
        if depth < 2: raise ValueError("depth must be >= 2")
        layers: list[nn.Module] = []
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width, bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(width, num_classes, bias=False))
        self.net = nn.Sequential(*layers)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _exclude_head(depth: int) -> list[str]:
    return [rf"net\.{2 * (depth - 1)}$"]

def _sync(d: torch.device) -> None:
    if d.type == "cuda": torch.cuda.synchronize(d)

@torch.no_grad()
def _agreement(model: nn.Module, feats: torch.Tensor, ref: torch.Tensor, *, bs: int) -> float:
    model.eval()
    correct = total = 0
    for s in range(0, feats.shape[0], bs):
        e = min(s + bs, feats.shape[0])
        pred = model(feats[s:e].to(dtype=torch.float16)).argmax(dim=1)
        correct += int((pred == ref[s:e]).sum().item()); total += e - s
    return 100.0 * float(correct) / float(max(total, 1))

@torch.no_grad()
def _lat(model: nn.Module, samp: torch.Tensor, *, w: int, r: int, dev: torch.device) -> float:
    model.eval()
    for _ in range(w): _ = model(samp)
    _sync(dev); t0 = time.perf_counter()
    for _ in range(r): _ = model(samp)
    _sync(dev)
    return 1000.0 * (time.perf_counter() - t0) / float(r)

def _med_lat(model: nn.Module, samp: torch.Tensor, *, cfg: BenchConfig, dev: torch.device) -> float:
    return float(statistics.median(
        [_lat(model, samp, w=cfg.warmup, r=cfg.repeats, dev=dev) for _ in range(cfg.timing_trials)]
    ))

def _counts(model: nn.Module) -> dict[str, int]:
    c = {"Linear": 0, "FP4WeightOnlyLinear": 0, "W4StorageInt8MmaLinear": 0}
    for m in model.modules():
        n = type(m).__name__
        if n in c: c[n] += 1
    return c

def _other_bytes(model: nn.Module, skip: tuple[type, ...]) -> int:
    t = 0
    for m in model.modules():
        if isinstance(m, skip): continue
        if any(isinstance(c, skip) for c in m.children()): continue
        for ten in list(m.parameters(recurse=False)) + list(m.buffers(recurse=False)):
            if ten is not None: t += int(ten.nbytes)
    return t

def _storage(model: nn.Module) -> int:
    w4 = [m for m in model.modules() if isinstance(m, W4StorageInt8MmaLinear)]
    fp4 = [m for m in model.modules() if isinstance(m, FP4WeightOnlyLinear)]
    if w4: return sum(m.storage_nbytes() for m in w4) + _other_bytes(model, (W4StorageInt8MmaLinear,))
    if fp4:
        t = 0
        for m in fp4: t += int(m.packed_weight.nbytes) + int(m.weight_scale.nbytes) + (int(m.bias.nbytes) if m.bias is not None else 0)
        return t + _other_bytes(model, (FP4WeightOnlyLinear,))
    return sum(int(t.nbytes) for t in list(model.parameters()) + list(model.buffers()))

def _pack_w4(src: nn.Module, cfg: BenchConfig) -> nn.Module:
    return quantize_with_fp4_weight_only(
        copy.deepcopy(src),
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": _exclude_head(cfg.depth), "group_size": cfg.group_size},
        inplace=True,
    ).model

def _prepare_tilelang(packed_w4: nn.Module, cfg: BenchConfig, calib: torch.Tensor) -> tuple[nn.Module, dict[str, float]]:
    b = quantize_with_w4_storage_int8_mma(
        copy.deepcopy(packed_w4),
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": _exclude_head(cfg.depth)},
        engine=cfg.engine,
        fallback_engine="torch_int_mm",
        source="fp4_weight_only",
        inplace=True,
        group_size=cfg.group_size,
        cache_int8_compute_view=True,
        activation_scale_mode="dynamic",
        block_m=cfg.block_m,
        block_n=cfg.block_n,
        block_k=cfg.block_k,
        threads=cfg.threads,
        num_stages=cfg.num_stages,
    ).model
    for m in b.modules():
        if isinstance(m, W4StorageInt8MmaLinear):
            m.output_dtype = torch.float16
            m.block_m = cfg.block_m
            m.block_n = cfg.block_n
            m.block_k = cfg.block_k
            m.threads = cfg.threads
            m.num_stages = cfg.num_stages
            m.release_int8_compute_view()
    am: dict[str, float] = {}; hs = []
    def _h(n: str):
        def _f(_m, inp, _o): v = float(inp[0].detach().float().abs().amax().item()); am[n] = max(am.get(n, 0.0), v)
        return _f
    for n, m in b.named_modules():
        if isinstance(m, W4StorageInt8MmaLinear): hs.append(m.register_forward_hook(_h(n)))
    with torch.no_grad():
        for _ in range(5): _ = b(calib)
    for h in hs: h.remove()
    sc: dict[str, float] = {}
    for n, m in b.named_modules():
        if not isinstance(m, W4StorageInt8MmaLinear): continue
        s = max(am.get(n, 1.0) / 127.0, 1e-6); sc[n] = s
        m.activation_scale_mode = "static"
        m._activation_scale = s
        m.release_int8_compute_view()
        c = m._ensure_compute_view()
        c.activation_scale_mode = "static"
        c.set_static_activation_scale(s)
        c.block_m = cfg.block_m
        c.block_n = cfg.block_n
        c.block_k = cfg.block_k
        c.threads = cfg.threads
        c.num_stages = cfg.num_stages
    with torch.no_grad():
        for _ in range(5): _ = b(calib)
    return b, sc

def run_benchmark(cfg: BenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BenchConfig()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    dev = torch.device("cuda"); _set_seed(cfg.seed); sm = torch.cuda.get_device_capability(dev)
    print(
        f"device={torch.cuda.get_device_name(dev)} sm_{sm[0]}{sm[1]} "
        f"d={cfg.depth} w={cfg.width} b={cfg.batch_size} engine={cfg.engine} "
        f"tile={cfg.block_m}x{cfg.block_n}x{cfg.block_k}"
    )
    ws = DeepMLP(depth=cfg.depth, width=cfg.width, num_classes=cfg.num_classes).to(dev, dtype=torch.float16).eval()
    pw4 = _pack_w4(ws, cfg).to(dev); pa = copy.deepcopy(pw4)
    feats = torch.randn(cfg.batch_size * cfg.num_batches, cfg.width, device=dev, dtype=torch.float16)
    samp = feats[:cfg.batch_size]
    with torch.no_grad(): rl = pa(feats).argmax(dim=1)
    a_acc = _agreement(pa, feats, rl, bs=cfg.batch_size)
    a_lat = _med_lat(pa, samp, cfg=cfg, dev=dev); a_st = _storage(pa)
    pb, ssc = _prepare_tilelang(pw4, cfg, calib=samp); pb = pb.to(dev); _ = pb(samp)
    b_acc = _agreement(pb, feats, rl, bs=cfg.batch_size)
    b_lat = _med_lat(pb, samp, cfg=cfg, dev=dev)
    for m in pb.modules():
        if isinstance(m, W4StorageInt8MmaLinear): m.release_int8_compute_view()
    b_st = _storage(pb); _ = pb(samp)
    drop = a_acc - b_acc; sp = a_lat / max(b_lat, 1e-9)
    pa_acc = drop <= cfg.max_accuracy_drop_pp; pa_sp = sp >= cfg.min_speedup_b_vs_a
    return {
        "device": torch.cuda.get_device_name(dev),
        "sm": f"sm_{sm[0]}{sm[1]}",
        "engine": cfg.engine,
        "config": {
            "depth": cfg.depth,
            "width": cfg.width,
            "batch_size": cfg.batch_size,
            "group_size": cfg.group_size,
            "static_scales": len(ssc),
            "no_qat": True,
            "block_m": cfg.block_m,
            "block_n": cfg.block_n,
            "block_k": cfg.block_k,
            "threads": cfg.threads,
            "num_stages": cfg.num_stages,
        },
        "path_a": {"meaning": "W4 deq->FP16", "acc": a_acc, "lat": a_lat, "storage": a_st, "linears": _counts(pa)},
        "path_b": {
            "meaning": f"W4+{cfg.engine} INT8",
            "acc": b_acc,
            "lat": b_lat,
            "storage": b_st,
            "linears": _counts(pb),
        },
        "delta": {"drop_pp": drop, "speedup": sp},
        "gates": {"max_drop": cfg.max_accuracy_drop_pp, "min_sp": cfg.min_speedup_b_vs_a, "acc": pa_acc, "sp": pa_sp, "overall": pa_acc and pa_sp},
    }

def main():
    cfg = BenchConfig()
    r = run_benchmark(cfg)
    print("=== W4 -> INT8 MMA acceptance (peak-oriented defaults) ===")
    a = r["path_a"]
    b = r["path_b"]
    print(
        f"A fp16-degrade: acc={a['acc']:.3f}% lat={a['lat']:.3f}ms "
        f"storage={a['storage']}B linears={a['linears']}"
    )
    print(
        f"B {r['engine']}: acc={b['acc']:.3f}% lat={b['lat']:.3f}ms "
        f"storage={b['storage']}B linears={b['linears']}"
    )
    print(f"delta: drop={r['delta']['drop_pp']:.3f}pp speedup={r['delta']['speedup']:.3f}x")
    g = r["gates"]
    print(
        f"gates: acc({'PASS' if g['acc'] else 'FAIL'}) "
        f"sp({'PASS' if g['sp'] else 'FAIL'}) "
        f"overall({'PASS' if g['overall'] else 'FAIL'})"
    )
    if not g["overall"]:
        raise SystemExit(1)

if __name__ == "__main__": main()
