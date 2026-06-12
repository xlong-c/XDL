from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

Backend = Literal["auto", "torch", "tilelang"]


@dataclass(frozen=True)
class RWKV8RosaDemoConfig:
    batch_size: int = 2
    seq_len: int = 12
    channels: int = 16
    vocab_size: int = 64
    num_layers: int = 2
    backend: Backend = "auto"
    tilelang_threads: int = 1
    tilelang_target_arch: str | None = None


def _check_symbol_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"q, k, v must share shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if q.dim() != 2:
        raise ValueError(f"expected [rows, seq_len] symbols, got {tuple(q.shape)}")
    if q.dtype != torch.uint8 or k.dtype != torch.uint8 or v.dtype != torch.uint8:
        raise ValueError(f"expected uint8 symbols, got {q.dtype}, {k.dtype}, {v.dtype}")


def _check_miss_value(miss_value: int) -> None:
    if miss_value < 0 or miss_value > 255:
        raise ValueError(f"miss_value must fit uint8, got {miss_value}")


def _resolve_backend(x: torch.Tensor, backend: Backend) -> Literal["torch", "tilelang"]:
    if backend == "torch":
        return "torch"
    if backend == "tilelang":
        return "tilelang"
    if backend != "auto":
        raise ValueError(f"unknown backend: {backend}")
    return "tilelang" if x.is_cuda else "torch"


def _require_tilelang() -> tuple[object, object]:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError(
            "TileLang is required for backend='tilelang'. Install project extras or use backend='torch'."
        ) from exc

    tilelang.set_log_level("WARNING")
    return tilelang, T


@torch.no_grad()
def rosa_suffix_match_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    miss_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference for the public RWKV8/ROSA suffix-memory idea.

    q, k, v are discrete symbols with shape [rows, seq_len]. For every row and
    time t, this searches the longest suffix of q[0:t+1] that appears in
    k[0:t]. If a match ends at history position j, the output symbol is
    v[j + 1]. Ties use the latest matching end position, matching the usual
    "rightmost occurrence" state kept by suffix automata.

    This is intentionally O(rows * T^4) in Python so the semantics are obvious.
    The TileLang kernel below keeps the same brute-force semantics but moves
    each row/time search to the GPU for teaching.
    """
    _check_symbol_inputs(q, k, v)
    _check_miss_value(miss_value)

    device = q.device
    q_cpu = q.detach().cpu().contiguous()
    k_cpu = k.detach().cpu().contiguous()
    v_cpu = v.detach().cpu().contiguous()
    rows, seq_len = q_cpu.shape

    out = torch.empty_like(q_cpu)
    hit = torch.zeros_like(q_cpu)

    for row in range(rows):
        for t in range(seq_len):
            best_len = 0
            best_end = -1
            for q_start in range(t + 1):
                suffix_len = t - q_start + 1
                for k_start in range(q_start):
                    matched = True
                    for offset in range(suffix_len):
                        if int(q_cpu[row, q_start + offset]) != int(k_cpu[row, k_start + offset]):
                            matched = False
                            break
                    if not matched:
                        continue

                    end_pos = k_start + suffix_len - 1
                    if suffix_len > best_len or (
                        suffix_len == best_len and end_pos > best_end
                    ):
                        best_len = suffix_len
                        best_end = end_pos

            if best_len > 0:
                out[row, t] = v_cpu[row, best_end + 1]
                hit[row, t] = 1
            else:
                out[row, t] = miss_value

    return out.to(device=device), hit.to(device=device)


@functools.cache
def build_rosa_suffix_match_kernel(
    seq_len: int,
    miss_value: int = 0,
    threads: int = 1,
    target_arch: str | None = None,
):
    """
    Build a scalar TileLang teaching kernel for ROSA suffix matching.

    Grid layout:
    - block x = row in the flattened [batch * channel_or_group] dimension.
    - block y = time position.

    One CUDA program computes one output symbol. This is deliberately simple
    and slow, because the tutorial focuses on the RWKV8/ROSA semantics before
    the production suffix-automaton optimization.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    _check_miss_value(miss_value)
    if threads <= 0:
        raise ValueError(f"threads must be positive, got {threads}")
    if target_arch is not None and not target_arch.startswith("sm_"):
        raise ValueError(f"target_arch must look like 'sm_80', got {target_arch}")

    tilelang, T = _require_tilelang()
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    target = {"kind": "cuda", "arch": target_arch} if target_arch is not None else None

    @tilelang.jit(pass_configs=pass_configs, target=target)
    def _factory():
        rows = T.symbolic("rows")

        @T.prim_func
        def kernel(
            q: T.Tensor((rows, seq_len), T.uint8),
            k: T.Tensor((rows, seq_len), T.uint8),
            v: T.Tensor((rows, seq_len), T.uint8),
            out: T.Tensor((rows, seq_len), T.uint8),
            hit: T.Tensor((rows, seq_len), T.uint8),
        ):
            with T.Kernel(rows, seq_len, threads=threads) as (row, t):
                best_len = T.alloc_var(dtype=T.int32)
                best_end = T.alloc_var(dtype=T.int32)

                best_len = 0
                best_end = -1
                hit[row, t] = 0
                out[row, t] = miss_value

                for q_start in T.serial(seq_len):
                    if q_start <= t:
                        suffix_len: T.int32 = t - q_start + 1
                        for k_start in T.serial(seq_len):
                            if k_start < q_start:
                                matched = T.alloc_var(dtype=T.int32)
                                matched = 1

                                for offset in T.serial(seq_len):
                                    if offset < suffix_len:
                                        matched = T.if_then_else(
                                            q[row, q_start + offset] == k[row, k_start + offset],
                                            matched,
                                            0,
                                        )

                                if matched == 1:
                                    end_pos: T.int32 = k_start + suffix_len - 1
                                    if suffix_len > best_len:
                                        best_len = suffix_len
                                        best_end = end_pos
                                    elif suffix_len == best_len and end_pos > best_end:
                                        best_end = end_pos

                if best_len > 0:
                    out[row, t] = v[row, best_end + 1]
                    hit[row, t] = 1

        return kernel

    return _factory()


@torch.no_grad()
def rosa_suffix_match_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    miss_value: int = 0,
    threads: int = 1,
    target_arch: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_symbol_inputs(q, k, v)
    _check_miss_value(miss_value)
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("TileLang backend requires CUDA tensors")

    q_contiguous = q.contiguous()
    k_contiguous = k.contiguous()
    v_contiguous = v.contiguous()
    out = torch.empty_like(q_contiguous)
    hit = torch.empty_like(q_contiguous)
    kernel = build_rosa_suffix_match_kernel(
        seq_len=q_contiguous.shape[1],
        miss_value=miss_value,
        threads=threads,
        target_arch=target_arch,
    )
    kernel(q_contiguous, k_contiguous, v_contiguous, out, hit)
    return out, hit


def rosa_suffix_match(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    miss_value: int = 0,
    backend: Backend = "auto",
    tilelang_threads: int = 1,
    tilelang_target_arch: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_backend = _resolve_backend(q, backend)
    if selected_backend == "tilelang":
        return rosa_suffix_match_tilelang(
            q=q,
            k=k,
            v=v,
            miss_value=miss_value,
            threads=tilelang_threads,
            target_arch=tilelang_target_arch,
        )
    return rosa_suffix_match_ref(q=q, k=k, v=v, miss_value=miss_value)


def binarize_channels(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [batch, seq_len, channels], got {tuple(x.shape)}")
    return (x > 0).to(torch.uint8)


def flatten_channel_symbols(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [batch, seq_len, channels], got {tuple(x.shape)}")
    return x.transpose(1, 2).reshape(-1, x.shape[1]).contiguous()


def restore_channel_symbols(x: torch.Tensor, batch_size: int, channels: int) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"expected [batch * channels, seq_len], got {tuple(x.shape)}")
    seq_len = x.shape[1]
    return x.view(batch_size, channels, seq_len).transpose(1, 2).contiguous()


def pack_bit_symbols(x: torch.Tensor, bits_per_symbol: int = 4) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [batch, seq_len, channels], got {tuple(x.shape)}")
    if bits_per_symbol <= 0 or bits_per_symbol > 8:
        raise ValueError(f"bits_per_symbol must be in [1, 8], got {bits_per_symbol}")
    batch_size, seq_len, channels = x.shape
    if channels % bits_per_symbol != 0:
        raise ValueError(
            f"channels must be divisible by bits_per_symbol, got {channels} and {bits_per_symbol}"
        )

    bits = binarize_channels(x).view(batch_size, seq_len, channels // bits_per_symbol, bits_per_symbol)
    weights = (2 ** torch.arange(bits_per_symbol, device=x.device, dtype=torch.int64)).view(
        1,
        1,
        1,
        bits_per_symbol,
    )
    symbols = (bits.to(torch.int64) * weights).sum(dim=-1).to(torch.uint8)
    return symbols


def unpack_bit_symbols(
    symbols: torch.Tensor,
    bits_per_symbol: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if symbols.dim() != 3:
        raise ValueError(f"expected [batch, seq_len, groups], got {tuple(symbols.shape)}")
    bit_ids = torch.arange(bits_per_symbol, device=symbols.device, dtype=torch.int64).view(
        1,
        1,
        1,
        bits_per_symbol,
    )
    bits = ((symbols.to(torch.int64).unsqueeze(-1) >> bit_ids) & 1).to(dtype)
    signed = bits * 2.0 - 1.0
    return signed.reshape(symbols.shape[0], symbols.shape[1], symbols.shape[2] * bits_per_symbol)


class RWKV8Rosa1Bit(nn.Module):
    """
    Educational RWKV8/ROSA 1-bit layer.

    The discrete suffix lookup is non-differentiable. Gradients flow through
    the learnable embedding scale and downstream layers, which is enough for
    studying data flow and kernel integration. A training-grade operator would
    need a custom estimator or the official training recipe.
    """

    def __init__(
        self,
        channels: int,
        backend: Backend = "auto",
        tilelang_threads: int = 1,
        tilelang_target_arch: str | None = None,
        zero_on_miss: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.backend = backend
        self.tilelang_threads = tilelang_threads
        self.tilelang_target_arch = tilelang_target_arch
        self.zero_on_miss = zero_on_miss
        self.emb = nn.Parameter(torch.ones(1, 1, channels))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_hit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                f"q, k, v must share shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
            )
        if q.dim() != 3:
            raise ValueError(f"expected [batch, seq_len, channels], got {tuple(q.shape)}")
        if q.shape[-1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {q.shape[-1]}")

        batch_size, _, channels = q.shape
        q_bits = flatten_channel_symbols(binarize_channels(q))
        k_bits = flatten_channel_symbols(binarize_channels(k))
        v_bits = flatten_channel_symbols(binarize_channels(v))
        symbols, hit = rosa_suffix_match(
            q_bits,
            k_bits,
            v_bits,
            miss_value=0,
            backend=self.backend,
            tilelang_threads=self.tilelang_threads,
            tilelang_target_arch=self.tilelang_target_arch,
        )
        restored = restore_channel_symbols(symbols, batch_size=batch_size, channels=channels)
        restored_hit = restore_channel_symbols(hit, batch_size=batch_size, channels=channels)
        out = (restored.to(dtype=q.dtype) * 2.0 - 1.0) * self.emb.to(dtype=q.dtype)
        if self.zero_on_miss:
            out = out * restored_hit.to(dtype=q.dtype)
        if return_hit:
            return out, restored_hit
        return out


class RWKV8Rosa4Bit(nn.Module):
    """
    Educational RWKV8/ROSA 4-bit layer.

    Every group of four channels is packed into one uint8 symbol in [0, 15],
    then the same suffix-memory lookup is applied per group. Misses are
    zeroed, following the public 4-bit demo convention.
    """

    def __init__(
        self,
        channels: int,
        bits_per_symbol: int = 4,
        backend: Backend = "auto",
        tilelang_threads: int = 1,
        tilelang_target_arch: str | None = None,
        zero_on_miss: bool = True,
    ) -> None:
        super().__init__()
        if channels % bits_per_symbol != 0:
            raise ValueError(
                f"channels must be divisible by bits_per_symbol, got {channels} and {bits_per_symbol}"
            )
        self.channels = channels
        self.bits_per_symbol = bits_per_symbol
        self.backend = backend
        self.tilelang_threads = tilelang_threads
        self.tilelang_target_arch = tilelang_target_arch
        self.zero_on_miss = zero_on_miss
        self.emb = nn.Parameter(torch.ones(1, 1, channels))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_hit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                f"q, k, v must share shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
            )
        if q.dim() != 3:
            raise ValueError(f"expected [batch, seq_len, channels], got {tuple(q.shape)}")
        if q.shape[-1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {q.shape[-1]}")

        batch_size, seq_len, channels = q.shape
        groups = channels // self.bits_per_symbol
        q_symbols = pack_bit_symbols(q, self.bits_per_symbol)
        k_symbols = pack_bit_symbols(k, self.bits_per_symbol)
        v_symbols = pack_bit_symbols(v, self.bits_per_symbol)

        q_flat = q_symbols.transpose(1, 2).reshape(-1, seq_len).contiguous()
        k_flat = k_symbols.transpose(1, 2).reshape(-1, seq_len).contiguous()
        v_flat = v_symbols.transpose(1, 2).reshape(-1, seq_len).contiguous()
        symbols, hit = rosa_suffix_match(
            q_flat,
            k_flat,
            v_flat,
            miss_value=0,
            backend=self.backend,
            tilelang_threads=self.tilelang_threads,
            tilelang_target_arch=self.tilelang_target_arch,
        )
        restored_symbols = symbols.view(batch_size, groups, seq_len).transpose(1, 2).contiguous()
        restored_hit = hit.view(batch_size, groups, seq_len).transpose(1, 2).contiguous()
        unpacked = unpack_bit_symbols(restored_symbols, self.bits_per_symbol, q.dtype)
        out = unpacked * self.emb.to(dtype=q.dtype)
        if self.zero_on_miss:
            hit_channels = (
                restored_hit.to(dtype=q.dtype)
                .unsqueeze(-1)
                .expand(batch_size, seq_len, groups, self.bits_per_symbol)
                .reshape(batch_size, seq_len, channels)
            )
            out = out * hit_channels
        if return_hit:
            return out, restored_hit
        return out


def time_shift(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [batch, seq_len, channels], got {tuple(x.shape)}")
    zero = torch.zeros_like(x[:, :1, :])
    return torch.cat([zero, x[:, :-1, :]], dim=1)


class RWKV8RosaMix(nn.Module):
    def __init__(self, channels: int, rosa_layer: nn.Module) -> None:
        super().__init__()
        self.channels = channels
        self.x_q = nn.Parameter(torch.zeros(1, 1, channels))
        self.x_k = nn.Parameter(torch.zeros(1, 1, channels))
        self.x_v = nn.Parameter(torch.zeros(1, 1, channels))
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.rosa = rosa_layer
        self.out_proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted_delta = time_shift(x) - x
        q = self.q_proj(x + shifted_delta * self.x_q)
        k = self.k_proj(x + shifted_delta * self.x_k)
        v = self.v_proj(x + shifted_delta * self.x_v)
        y = self.rosa(q, k, v)
        if isinstance(y, tuple):
            y = y[0]
        return self.out_proj(y)


class RWKV8SquaredReluFFN(nn.Module):
    def __init__(self, channels: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden = channels * hidden_mult
        self.x_k = nn.Parameter(torch.zeros(1, 1, channels))
        self.key = nn.Linear(channels, hidden, bias=False)
        self.value = nn.Linear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted_delta = time_shift(x) - x
        k = x + shifted_delta * self.x_k
        return self.value(F.relu(self.key(k)).square())


class RWKV8PureRosaBlock(nn.Module):
    def __init__(self, channels: int, rosa_layer: nn.Module, hidden_mult: int = 4) -> None:
        super().__init__()
        self.ln_rosa = nn.LayerNorm(channels)
        self.ln_ffn = nn.LayerNorm(channels)
        self.rosa = RWKV8RosaMix(channels, rosa_layer)
        self.ffn = RWKV8SquaredReluFFN(channels, hidden_mult=hidden_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.rosa(self.ln_rosa(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x


class TinyRWKV8RosaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        channels: int,
        num_layers: int,
        rosa_bits: Literal[1, 4] = 1,
        backend: Backend = "auto",
        tilelang_threads: int = 1,
        tilelang_target_arch: str | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, channels)
        self.blocks = nn.ModuleList(
            [
                RWKV8PureRosaBlock(
                    channels=channels,
                    rosa_layer=self._make_rosa_layer(
                        channels=channels,
                        rosa_bits=rosa_bits,
                        backend=backend,
                        tilelang_threads=tilelang_threads,
                        tilelang_target_arch=tilelang_target_arch,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_out = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, vocab_size, bias=False)

    @staticmethod
    def _make_rosa_layer(
        channels: int,
        rosa_bits: Literal[1, 4],
        backend: Backend,
        tilelang_threads: int,
        tilelang_target_arch: str | None,
    ) -> nn.Module:
        if rosa_bits == 1:
            return RWKV8Rosa1Bit(
                channels=channels,
                backend=backend,
                tilelang_threads=tilelang_threads,
                tilelang_target_arch=tilelang_target_arch,
            )
        if rosa_bits == 4:
            return RWKV8Rosa4Bit(
                channels=channels,
                backend=backend,
                tilelang_threads=tilelang_threads,
                tilelang_target_arch=tilelang_target_arch,
            )
        raise ValueError(f"unsupported rosa_bits: {rosa_bits}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"expected [batch, seq_len] token ids, got {tuple(tokens.shape)}")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_out(x))


def demo_reference_semantics() -> None:
    q = torch.tensor([[1, 0, 1, 0, 1, 1, 0, 1]], dtype=torch.uint8)
    k = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 1]], dtype=torch.uint8)
    v = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 0]], dtype=torch.uint8)
    out, hit = rosa_suffix_match_ref(q, k, v)
    print("reference q :", q.tolist()[0])
    print("reference k :", k.tolist()[0])
    print("reference v :", v.tolist()[0])
    print("rosa out    :", out.tolist()[0])
    print("hit mask    :", hit.tolist()[0])


def demo_tiny_lm(config: RWKV8RosaDemoConfig) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRWKV8RosaLM(
        vocab_size=config.vocab_size,
        channels=config.channels,
        num_layers=config.num_layers,
        rosa_bits=1,
        backend=config.backend,
        tilelang_threads=config.tilelang_threads,
        tilelang_target_arch=config.tilelang_target_arch,
    ).to(device)
    tokens = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.seq_len),
        device=device,
    )
    logits = model(tokens)
    print(f"tiny lm logits: shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}")


def demo_tilelang_vs_reference(config: RWKV8RosaDemoConfig) -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skip TileLang ROSA kernel demo")
        return

    torch.manual_seed(1)
    q = torch.randint(0, 2, (config.batch_size * config.channels, config.seq_len), device="cuda", dtype=torch.uint8)
    k = torch.randint(0, 2, q.shape, device="cuda", dtype=torch.uint8)
    v = torch.randint(0, 2, q.shape, device="cuda", dtype=torch.uint8)
    out_ref, hit_ref = rosa_suffix_match_ref(q, k, v)
    out_tl, hit_tl = rosa_suffix_match_tilelang(
        q,
        k,
        v,
        threads=config.tilelang_threads,
        target_arch=config.tilelang_target_arch,
    )
    torch.testing.assert_close(out_tl.cpu(), out_ref.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(hit_tl.cpu(), hit_ref.cpu(), rtol=0, atol=0)
    print("TileLang ROSA kernel matches CPU reference")


def main() -> None:
    config = RWKV8RosaDemoConfig()
    demo_reference_semantics()
    demo_tiny_lm(config)
    demo_tilelang_vs_reference(config)


if __name__ == "__main__":
    main()
