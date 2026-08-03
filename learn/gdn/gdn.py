from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


RuleOutput = Tuple[torch.Tensor, Optional[torch.Tensor]]


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _validate_rule_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, value must all have shape [batch, seq, heads, dim].")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g and beta must both have shape [batch, seq, heads].")
    if query.shape != key.shape:
        raise ValueError("query and key must have the same shape.")
    if query.shape[:3] != value.shape[:3]:
        raise ValueError("value must match query/key on [batch, seq, heads].")
    if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError("g and beta must match query/key on [batch, seq, heads].")
    if initial_state is not None:
        expected_state = (
            query.shape[0],
            query.shape[2],
            query.shape[3],
            value.shape[3],
        )
        if tuple(initial_state.shape) != expected_state:
            raise ValueError(
                "initial_state must have shape "
                f"{expected_state}, got {tuple(initial_state.shape)}."
            )


def gated_delta_rule_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
) -> RuleOutput:
    """
    Educational recurrent Gated Delta Rule.

    Shapes:
        query/key: [B, T, H, Dk]
        value:     [B, T, H, Dv]
        g/beta:    [B, T, H]
        state:     [B, H, Dk, Dv]
    """
    _validate_rule_inputs(query, key, value, g, beta, initial_state)

    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # Work in [B, H, T, D] so each head carries its own temporal state.
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, key_dim = key.shape
    value_dim = value.shape[-1]
    # Match the usual attention-style scaling before reading from state.
    scale = 1.0 / math.sqrt(query.shape[-1])
    query = query * scale

    output = torch.zeros(
        batch_size,
        num_heads,
        sequence_length,
        value_dim,
        dtype=value.dtype,
        device=value.device,
    )
    # `state` stores one key-to-value memory matrix per head: [B, H, Dk, Dv].
    state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )

    for index in range(sequence_length):
        q_t = query[:, :, index]
        k_t = key[:, :, index]
        v_t = value[:, :, index]
        decay_t = g[:, :, index].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, index].unsqueeze(-1)

        # `state` is stored as [Dk, Dv], so reading with the current key is `state^T k_t`.
        # First forget part of the old memory, then read what the old state predicts.
        state = state * decay_t
        cached_value = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - cached_value) * beta_t
        # Delta rule writes a rank-1 correction along the current key direction.
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # The current token reads from the updated state, not the previous one.
        output[:, :, index] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    final_state = state if output_final_state else None
    return output.transpose(1, 2).contiguous().to(initial_dtype), final_state


def gated_delta_rule_chunkwise(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
) -> RuleOutput:
    """
    Educational chunkwise Gated Delta Rule.

    This mirrors the official PyTorch fallback logic used in Hugging Face
    `Qwen3NextGatedDeltaNet`, keeping the math explicit instead of hiding it
    inside fused CUDA kernels.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    _validate_rule_inputs(query, key, value, g, beta, initial_state)

    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # Reorder to [B, H, T, D] to match the recurrent implementation and per-head state layout.
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, key_dim = key.shape
    value_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    # Pad to an integer number of chunks so we can reshape into [num_chunks, chunk_size].
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    scale = 1.0 / math.sqrt(query.shape[-1])
    query = query * scale

    # Fold beta into key/value once so later formulas match the delta-rule residual update.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Group tokens into chunks: [B, H, num_chunks, chunk_size, D].
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    lower_causal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # Prefix log-decays let us recover any in-chunk relative decay with simple differences.
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    # This lower-triangular operator materializes the chunk-local delta recursion.
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(lower_causal_mask, 0)
    for row_index in range(1, chunk_size):
        # Solve the causal lower-triangular dependency one row at a time inside the chunk.
        row = attn[..., row_index, :row_index].clone()
        sub = attn[..., :row_index, :row_index].clone()
        attn[..., row_index, :row_index] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    # These are the chunk-local contributions after unrolling the intra-chunk recursion.
    materialized_value = attn @ v_beta
    materialized_key = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    output = torch.zeros_like(value)

    num_chunks = total_sequence_length // chunk_size
    for chunk_index in range(num_chunks):
        q_i = query[:, :, chunk_index]
        k_i = key[:, :, chunk_index]
        v_i = materialized_value[:, :, chunk_index]
        g_i = g[:, :, chunk_index]
        decay_i = decay_mask[:, :, chunk_index]

        # `intra` handles dependencies inside the chunk, `inter` reads from carried state.
        intra = q_i @ k_i.transpose(-1, -2) * decay_i
        carried = materialized_key[:, :, chunk_index] @ state
        # Only the residual unexplained by the incoming cross-chunk state still needs writing.
        v_new = v_i - carried
        inter = (q_i * g_i.exp().unsqueeze(-1)) @ state
        output[:, :, chunk_index] = inter + intra @ v_new

        final_decay = g_i[:, :, -1, None, None].exp()
        chunk_decay = (g_i[:, :, -1, None] - g_i).exp()[..., None]
        # Collapse the whole chunk back into one outgoing state for the next chunk.
        state = state * final_decay + (k_i * chunk_decay).transpose(-1, -2) @ v_new

    final_state = state if output_final_state else None
    output = output.reshape(output.shape[0], output.shape[1], -1, output.shape[-1])
    output = output[:, :, :sequence_length]
    return output.transpose(1, 2).contiguous().to(initial_dtype), final_state


@dataclass
class GatedDeltaNetDemoConfig:
    seed: int = 0
    batch_size: int = 2
    seq_len: int = 12
    hidden_size: int = 32
    num_heads: int = 4
    head_dim: int = 8
    value_head_dim: int = 8
    chunk_size: int = 4


class SimpleGatedDeltaNet(nn.Module):
    """
    Small teaching block that exposes the GDN core without convolution,
    caching helpers, or mixed full-attention layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        head_dim: int = 16,
        value_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim or head_dim
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_heads * self.value_head_dim

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.output_gate_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.norm = nn.LayerNorm(self.value_dim)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _project(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = x.shape
        query = self.q_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = self.k_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = self.v_proj(x).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.value_head_dim,
        )
        # Beta controls how much of the residual gets written into memory at this step.
        beta = self.beta_proj(x).sigmoid()
        # Parameterize decay in negative log-space so exp(g) stays in (0, 1].
        g = -self.A_log.float().exp().view(1, 1, -1) * F.softplus(
            self.g_proj(x).float() + self.dt_bias.view(1, 1, -1)
        )
        # Output gating is applied after the core rule has produced the per-head value stream.
        z = F.silu(self.output_gate_proj(x)).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.value_head_dim,
        )
        return query, key, value, g, beta, z

    def forward(
        self,
        x: torch.Tensor,
        mode: Literal["recurrent", "chunkwise"] = "recurrent",
        chunk_size: int = 64,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        query, key, value, g, beta, z = self._project(x)

        if mode == "recurrent":
            core, final_state = gated_delta_rule_recurrent(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=return_state,
                use_qk_l2norm=True,
            )
        elif mode == "chunkwise":
            core, final_state = gated_delta_rule_chunkwise(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=return_state,
                use_qk_l2norm=True,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # The rule output is normalized and gated before projecting back to hidden size.
        core = core.reshape(x.shape[0], x.shape[1], self.value_dim)
        core = self.norm(core)
        core = core.reshape(x.shape[0], x.shape[1], self.num_heads, self.value_head_dim)
        core = core * z.to(core.dtype)
        output = self.out_proj(core.reshape(x.shape[0], x.shape[1], self.value_dim))

        if return_state:
            return output, final_state
        return output


def demo_rule_consistency(config: GatedDeltaNetDemoConfig) -> None:
    torch.manual_seed(config.seed)
    model = SimpleGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        value_head_dim=config.value_head_dim,
    )
    x = torch.randn(config.batch_size, config.seq_len, config.hidden_size)

    recurrent_out, recurrent_state = model(
        x,
        mode="recurrent",
        return_state=True,
    )
    chunk_out, chunk_state = model(
        x,
        mode="chunkwise",
        chunk_size=config.chunk_size,
        return_state=True,
    )

    max_output_diff = (recurrent_out - chunk_out).abs().max().item()
    max_state_diff = 0.0
    if recurrent_state is not None and chunk_state is not None:
        max_state_diff = (recurrent_state - chunk_state).abs().max().item()

    print("GDN educational demo")
    print(f"batch={config.batch_size} seq={config.seq_len} hidden={config.hidden_size}")
    print(f"heads={config.num_heads} head_dim={config.head_dim} chunk_size={config.chunk_size}")
    print(f"max |recurrent - chunkwise| output diff: {max_output_diff:.6e}")
    print(f"max |recurrent - chunkwise| state diff:  {max_state_diff:.6e}")
    print("sample recurrent output[0, 0, :6]:", recurrent_out[0, 0, :6].tolist())


if __name__ == "__main__":
    demo_rule_consistency(GatedDeltaNetDemoConfig())
