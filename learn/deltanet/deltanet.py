import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class KimiLinearAttention(nn.Module):
    """
    KimiLinearAttention: Linear attention variant with kernel feature map.
    Based on linear attention principles: O = φ(Q) * (φ(K)^T * V) / (φ(Q) * (φ(K)^T * 1))
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, feature_map: str = 'elu'):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.feature_map = feature_map
        if feature_map == 'elu':
            self.phi = lambda x: F.elu(x) + 1.0
        elif feature_map == 'relu':
            self.phi = lambda x: F.relu(x)
        elif feature_map == 'softmax':
            self.phi = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unsupported feature map: {feature_map}")

        self.scale = dim_head ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) or (batch, seq_len, seq_len)
        Returns:
            out: (batch, seq_len, dim)
        """
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, self.dim_head).transpose(1, 2), qkv)

        # Apply feature map
        q = self.phi(q) * self.scale
        k = self.phi(k)

        # Compute linear attention
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        z = k.sum(dim=2, keepdim=True)

        # Compute output
        out = torch.einsum('b h n d, b h d e -> b h n e', q, kv)
        z_squeezed = z.squeeze(2)  # (batch, heads, dim_head)
        denominator = torch.einsum('b h n d, b h d -> b h n', q, z_squeezed) + 1e-8
        out = out / denominator.unsqueeze(-1)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)
            out = out * mask

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

    def prefill(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prefill stage: process full sequence and return output plus KV cache.

        Args:
            x: (batch, seq_len, dim)
            mask: optional mask

        Returns:
            out: (batch, seq_len, dim)
            cache: tuple of (kv_cache, z_cache) for decode stage
        """
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, self.dim_head).transpose(1, 2), qkv)

        # Apply feature map
        q = self.phi(q) * self.scale
        k = self.phi(k)

        # Compute linear attention and cache
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)  # (batch, heads, dim_head, dim_head)
        z = k.sum(dim=2, keepdim=True)  # (batch, heads, 1, dim_head)

        # Compute output
        out = torch.einsum('b h n d, b h d e -> b h n e', q, kv)
        z_squeezed = z.squeeze(2)  # (batch, heads, dim_head)
        denominator = torch.einsum('b h n d, b h d -> b h n', q, z_squeezed) + 1e-8
        out = out / denominator.unsqueeze(-1)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)
            out = out * mask

        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)

        # Cache kv and z for decode stage
        cache = (kv, z)
        return out, cache

    def decode(self, x: torch.Tensor, cache: Tuple[torch.Tensor, torch.Tensor],
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode stage: process single token with cached KV.

        Args:
            x: (batch, 1, dim) - single token
            cache: tuple of (kv_cache, z_cache) from prefill
            mask: optional mask

        Returns:
            out: (batch, 1, dim)
            new_cache: updated cache for next token
        """
        b, n, _ = x.shape  # n should be 1
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, self.dim_head).transpose(1, 2), qkv)

        # Apply feature map
        q = self.phi(q) * self.scale
        k = self.phi(k)

        kv_cache, z_cache = cache

        # Update cache with new k, v
        # kv_cache shape: (batch, heads, dim_head, dim_head)
        # k shape: (batch, heads, 1, dim_head), v shape: (batch, heads, 1, dim_head)
        kv_update = torch.einsum('b h d, b h e -> b h d e', k.squeeze(2), v.squeeze(2))
        new_kv = kv_cache + kv_update

        # Update z cache
        new_z = z_cache + k  # z_cache shape: (batch, heads, 1, dim_head)

        # Compute output using updated cache
        out = torch.einsum('b h n d, b h d e -> b h n e', q, new_kv)
        new_z_squeezed = new_z.squeeze(2)  # (batch, heads, dim_head)
        denominator = torch.einsum('b h n d, b h d -> b h n', q, new_z_squeezed) + 1e-8
        out = out / denominator.unsqueeze(-1)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)
            out = out * mask

        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)

        return out, (new_kv, new_z)


class DeltaNet(nn.Module):
    """
    DeltaNet: Attention mechanism with delta update rule.
    Combines linear attention with gated delta updates.
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 feature_map: str = 'elu', use_delta: bool = True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.use_delta = use_delta

        self.linear_attn = KimiLinearAttention(dim, heads, dim_head, feature_map)

        if use_delta:
            self.delta_gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            )
            self.delta_transform = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                prev_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: optional mask
            prev_state: previous hidden state for delta update (batch, dim)
        Returns:
            out: (batch, seq_len, dim)
            new_state: (batch, dim) if use_delta else None
        """
        b, n, d = x.shape

        # Linear attention
        attn_out = self.linear_attn(x, mask)

        if self.use_delta and prev_state is not None:
            # Delta update mechanism
            prev_state_expanded = prev_state.unsqueeze(1).expand(-1, n, -1)
            gate_input = torch.cat([attn_out, prev_state_expanded], dim=-1)
            gate = self.delta_gate(gate_input)

            delta = self.delta_transform(attn_out)
            delta_out = gate * delta + (1 - gate) * prev_state_expanded

            out = self.norm(x + delta_out)
            new_state = delta_out[:, -1, :]  # last token state
        else:
            out = self.norm(x + attn_out)
            new_state = out[:, -1, :] if self.use_delta else None

        return out, new_state

    def prefill(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Prefill stage: process full sequence and return output plus cache.

        Args:
            x: (batch, seq_len, dim)
            mask: optional mask

        Returns:
            out: (batch, seq_len, dim)
            cache: tuple of (attention_cache, delta_state) for decode stage
        """
        # Process through linear attention with prefill
        attn_out, attn_cache = self.linear_attn.prefill(x, mask)

        if self.use_delta:
            # No previous state for prefill, so just compute output
            out = self.norm(x + attn_out)
            delta_state = out[:, -1, :]  # last token state for delta update
            cache = (attn_cache, delta_state)
        else:
            out = self.norm(x + attn_out)
            cache = (attn_cache, None)

        return out, cache

    def decode(self, x: torch.Tensor, cache: Tuple,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Decode stage: process single token with cached values.

        Args:
            x: (batch, 1, dim) - single token
            cache: tuple of (attention_cache, delta_state) from prefill
            mask: optional mask

        Returns:
            out: (batch, 1, dim)
            new_cache: updated cache for next token
        """
        attn_cache, prev_state = cache

        # Process through linear attention with decode
        attn_out, new_attn_cache = self.linear_attn.decode(x, attn_cache, mask)

        if self.use_delta and prev_state is not None:
            # Delta update mechanism
            prev_state_expanded = prev_state.unsqueeze(1)  # (batch, 1, dim)
            gate_input = torch.cat([attn_out, prev_state_expanded], dim=-1)
            gate = self.delta_gate(gate_input)

            delta = self.delta_transform(attn_out)
            delta_out = gate * delta + (1 - gate) * prev_state_expanded

            out = self.norm(x + delta_out)
            new_delta_state = delta_out[:, -1, :]  # last token state
        else:
            out = self.norm(x + attn_out)
            new_delta_state = out[:, -1, :] if self.use_delta else None

        new_cache = (new_attn_cache, new_delta_state)
        return out, new_cache


def prefill(model: nn.Module, input_ids: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prefill stage: process full input sequence.

    Args:
        model: DeltaNet model
        input_ids: (batch, seq_len)
        mask: optional mask

    Returns:
        logits: (batch, seq_len, vocab_size)
        hidden_states: (batch, seq_len, hidden_dim)
    """
    # In a real implementation, this would include embedding lookup and multiple layers
    # For demonstration, we assume model is a DeltaNet layer
    batch_size, seq_len = input_ids.shape

    # Create dummy embeddings (replace with actual embedding layer)
    hidden_dim = 512  # example dimension
    embeddings = torch.randn(batch_size, seq_len, hidden_dim, device=input_ids.device)

    # Process through DeltaNet
    output, state = model(embeddings, mask)

    # Dummy projection to vocab (replace with actual LM head)
    vocab_size = 50257  # example vocab size
    logits = torch.randn(batch_size, seq_len, vocab_size, device=input_ids.device)

    return logits, output


def decode(model: nn.Module, input_ids: torch.Tensor,
           past_key_values: Optional[Tuple] = None,
           mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
    """
    Decode stage: autoregressive token generation.

    Args:
        model: DeltaNet model
        input_ids: (batch, 1) - single token
        past_key_values: cached previous states
        mask: optional mask

    Returns:
        logits: (batch, 1, vocab_size)
        new_key_values: updated cached states
    """
    batch_size = input_ids.shape[0]

    # Create dummy embeddings for the single token
    hidden_dim = 512
    embeddings = torch.randn(batch_size, 1, hidden_dim, device=input_ids.device)

    # Use previous state if available
    prev_state = past_key_values if past_key_values is not None else None

    # Process through DeltaNet
    output, new_state = model(embeddings, mask, prev_state)

    # Dummy projection to vocab
    vocab_size = 50257
    logits = torch.randn(batch_size, 1, vocab_size, device=input_ids.device)

    return logits, new_state


# Example usage
if __name__ == "__main__":
    # Test KimiLinearAttention forward
    attn = KimiLinearAttention(dim=512, heads=8, dim_head=64)
    x = torch.randn(2, 32, 512)
    out = attn(x)
    print(f"KimiLinearAttention forward output shape: {out.shape}")

    # Test KimiLinearAttention prefill and decode
    print("\n--- Testing KimiLinearAttention prefill/decycle ---")
    attn_prefill_out, attn_cache = attn.prefill(x)
    print(f"KimiLinearAttention prefill output shape: {attn_prefill_out.shape}")
    print(f"KimiLinearAttention cache type: {type(attn_cache)}, length: {len(attn_cache)}")

    # Test decode with single token
    x_single = torch.randn(2, 1, 512)
    attn_decode_out, new_cache = attn.decode(x_single, attn_cache)
    print(f"KimiLinearAttention decode output shape: {attn_decode_out.shape}")
    print(f"KimiLinearAttention new cache type: {type(new_cache)}, length: {len(new_cache)}")

    # Test DeltaNet forward
    deltanet = DeltaNet(dim=512, heads=8, dim_head=64, use_delta=True)
    x = torch.randn(2, 32, 512)
    out, state = deltanet(x)
    print(f"\nDeltaNet forward output shape: {out.shape}, state shape: {state.shape}")

    # Test DeltaNet prefill and decode
    print("\n--- Testing DeltaNet prefill/decycle ---")
    deltanet_prefill_out, deltanet_cache = deltanet.prefill(x)
    print(f"DeltaNet prefill output shape: {deltanet_prefill_out.shape}")
    print(f"DeltaNet cache type: {type(deltanet_cache)}, length: {len(deltanet_cache)}")

    # Test decode with single token
    x_single = torch.randn(2, 1, 512)
    deltanet_decode_out, deltanet_new_cache = deltanet.decode(x_single, deltanet_cache)
    print(f"DeltaNet decode output shape: {deltanet_decode_out.shape}")
    print(f"DeltaNet new cache type: {type(deltanet_new_cache)}, length: {len(deltanet_new_cache)}")

    # Test compatibility with old prefill/decode functions (optional)
    print("\n--- Testing old prefill/decode functions (for backward compatibility) ---")
    dummy_input = torch.randint(0, 50257, (2, 32))
    logits, hidden = prefill(deltanet, dummy_input)
    print(f"Old prefill logits shape: {logits.shape}, hidden shape: {hidden.shape}")

    single_token = torch.randint(0, 50257, (2, 1))
    logits, new_state = decode(deltanet, single_token)
    print(f"Old decode logits shape: {logits.shape}, new_state shape: {new_state.shape}")