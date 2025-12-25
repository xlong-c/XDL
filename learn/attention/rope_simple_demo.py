"""
RoPE (Rotary Position Embedding) Simple Demo
=============================================

Simplified demonstration of RoPE for educational purposes
"""

import torch
import torch.nn.functional as F
import math


def compute_rope_frequencies(dim, base=10000):
    """
    Compute frequencies for RoPE
    """
    # Create frequencies for each dimension
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return freqs


def apply_rope(x, position_ids=None):
    """
    Apply RoPE to input tensor

    Args:
        x: input tensor [..., dim]
        position_ids: position indices [...]

    Returns:
        RoPE-applied tensor
    """
    batch_size, seq_len, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"

    # Compute positions
    if position_ids is None:
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Compute frequencies
    freqs = compute_rope_frequencies(dim).to(x.device)

    # Compute angles: [batch_size, seq_len, dim//2]
    angles = torch.einsum('bs,d->bsd', position_ids.float(), freqs)

    # Compute sin and cos: [batch_size, seq_len, dim//2]
    cos = torch.cos(angles).unsqueeze(-1)
    sin = torch.sin(angles).unsqueeze(-1)

    # Split input into real and imaginary parts
    x_real = x[..., 0::2]  # [batch_size, seq_len, dim//2]
    x_imag = x[..., 1::2]  # [batch_size, seq_len, dim//2]

    # Apply rotation formula
    x_rot_real = x_real * cos.squeeze(-1) - x_imag * sin.squeeze(-1)
    x_rot_imag = x_real * sin.squeeze(-1) + x_imag * cos.squeeze(-1)

    # Interleave and combine
    x_rot = torch.zeros_like(x)
    x_rot[..., 0::2] = x_rot_real
    x_rot[..., 1::2] = x_rot_imag

    return x_rot, cos.squeeze(-1), sin.squeeze(-1)


def traditional_positional_encoding(x, max_len=5000):
    """
    Traditional sinusoidal positional encoding
    """
    batch_size, seq_len, d_model = x.shape

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return x + pe[:seq_len].unsqueeze(0)


def demo_rope_basics():
    """Demonstrate basic RoPE functionality"""
    print("="*60)
    print("RoPE Basic Functionality Demo")
    print("="*60)

    # Parameters
    batch_size = 2
    seq_len = 10
    dim = 64

    # Create random input
    x = torch.randn(batch_size, seq_len, dim)
    print(f"Input tensor shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")

    # Apply RoPE
    x_rope, cos, sin = apply_rope(x)

    print(f"RoPE output shape: {x_rope.shape}")
    print(f"RoPE output range: [{x_rope.min():.3f}, {x_rope.max():.3f}]")
    print(f"Cos range: [{cos.min():.3f}, {cos.max():.3f}]")
    print(f"Sin range: [{sin.min():.3f}, {sin.max():.3f}]")

    # Show position encoding patterns
    print("\nPosition encoding patterns:")
    print("Position | cos[0] | sin[0] | cos[1] | sin[1] | cos[2] | sin[2]")
    print("-" * 55)
    for pos in [0, 1, 2, 5, 9]:
        print(f"{pos:4d} | {cos[0, pos, 0]:6.3f} | {sin[0, pos, 0]:6.3f} | "
              f"{cos[0, pos, 1]:6.3f} | {sin[0, pos, 1]:6.3f} | "
              f"{cos[0, pos, 2]:6.3f} | {sin[0, pos, 2]:6.3f}")


def demo_position_preservation():
    """Demonstrate RoPE's position preservation property"""
    print("\n" + "="*60)
    print("RoPE Position Preservation Demo")
    print("="*60)

    # Create two identical vectors at different positions
    batch_size = 1
    seq_len = 15
    dim = 32

    # Create a base vector
    base_vector = torch.randn(dim)

    # Create tensor with the same vector at different positions
    x = torch.zeros(batch_size, seq_len, dim)
    x[0, 3] = base_vector  # Position 3
    x[0, 7] = base_vector  # Position 7

    # Apply RoPE
    x_rope, _, _ = apply_rope(x)

    # Compute similarity between the two positions
    similarity = F.cosine_similarity(x_rope[0, 3], x_rope[0, 7], dim=0)

    print("Original vectors at positions 3 and 7 are identical")
    print("After RoPE transformation:")
    print(f"  Position 3 vector range: [{x_rope[0, 3].min():.3f}, {x_rope[0, 3].max():.3f}]")
    print(f"  Position 7 vector range: [{x_rope[0, 7].min():.3f}, {x_rope[0, 7].max():.3f}]")
    print(f"  Cosine similarity: {similarity:.6f}")

    # Test with different relative distances
    print("\nSimilarity across different relative distances:")
    print("Relative Distance | Cosine Similarity")
    print("-" * 35)

    for distance in [1, 2, 4, 8]:
        if distance < seq_len:
            pos1 = 5
            pos2 = pos1 + distance
            x_test = torch.zeros(batch_size, seq_len, dim)
            x_test[0, pos1] = base_vector
            x_test[0, pos2] = base_vector

            x_test_rope, _, _ = apply_rope(x_test)
            sim = F.cosine_similarity(x_test_rope[0, pos1], x_test_rope[0, pos2], dim=0)
            print(f"{distance:13d} | {sim:15.6f}")


def compare_with_traditional():
    """Compare RoPE with traditional positional encoding"""
    print("\n" + "="*60)
    print("RoPE vs Traditional Positional Encoding Comparison")
    print("="*60)

    batch_size = 1
    seq_len = 12
    dim = 48

    # Create input
    x = torch.randn(batch_size, seq_len, dim)

    # Traditional positional encoding
    x_trad = traditional_positional_encoding(x)

    # RoPE
    x_rope, _, _ = apply_rope(x)

    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Traditional PE range: [{x_trad.min():.3f}, {x_trad.max():.3f}]")
    print(f"RoPE range: [{x_rope.min():.3f}, {x_rope.max():.3f}]")

    # Compare position sensitivity
    print("\nPosition sensitivity analysis:")
    print("Position | Traditional Change | RoPE Change")
    print("-" * 42)

    for pos in range(seq_len):
        # Create perturbed input
        x_perturbed = x.clone()
        x_perturbed[0, pos] += 0.1

        # Apply encodings
        x_trad_perturbed = traditional_positional_encoding(x_perturbed)
        x_rope_perturbed, _, _ = apply_rope(x_perturbed)

        # Compute changes
        trad_change = torch.norm(x_trad_perturbed[0, pos] - x_trad[0, pos]).item()
        rope_change = torch.norm(x_rope_perturbed[0, pos] - x_rope[0, pos]).item()

        print(f"{pos:4d} | {trad_change:14.6f} | {rope_change:11.6f}")


def mathematical_explanation():
    """Mathematical explanation of RoPE"""
    print("\n" + "="*60)
    print("Mathematical Explanation of RoPE")
    print("="*60)

    print("1. Core Concept:")
    print("   RoPE encodes position through 2D rotation matrices")
    print("   Each position corresponds to a rotation angle θ")

    print("\n2. Rotation Formula:")
    print("   For a 2D vector [x, y] at position p:")
    print("   [x', y'] = [x*cos(θ) - y*sin(θ), x*sin(θ) + y*cos(θ)]")
    print("   where θ = p * 10000^(-2i/d)")

    print("\n3. Frequency Calculation:")
    print("   Frequencies decrease exponentially across dimensions")
    print("   This allows modeling of both local and global position relationships")

    print("\n4. Key Properties:")
    print("   • Relative position: dot(q₁, k₂) depends only on (p₂-p₁)")
    print("   • No additional parameters")
    print("   • Linear computational complexity")
    print("   • Handles variable length sequences")

    print("\n5. Application in Attention:")
    print("   q_rot = RoPE(q, position)")
    print("   k_rot = RoPE(k, position)")
    print("   attention = softmax(q_rot · k_rot^T / √d)")

    # Demonstrate the math with a simple example
    print("\n6. Simple Mathematical Example:")
    print("   Let's trace RoPE for a 2D vector [1, 0] at different positions:")

    dim = 4
    base = 10000
    positions = [0, 1, 2, 3]

    for pos in positions:
        # Compute frequency for first dimension
        freq = 1.0 / (base ** (0 / dim))
        angle = pos * freq
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)

        # Apply rotation to [1, 0]
        x_new = 1 * cos_val - 0 * sin_val
        y_new = 1 * sin_val + 0 * cos_val

        print(f"   Position {pos}: angle={angle:.3f}, result=[{x_new:.3f}, {y_new:.3f}]")


def performance_comparison():
    """Simple performance comparison"""
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)

    import time

    batch_size = 4
    seq_len = 256
    dim = 64
    num_iterations = 100

    # Create test data
    x = torch.randn(batch_size, seq_len, dim)

    # Test RoPE
    start_time = time.time()
    for _ in range(num_iterations):
        x_rope, _, _ = apply_rope(x)
    rope_time = time.time() - start_time

    # Test traditional PE
    start_time = time.time()
    for _ in range(num_iterations):
        traditional_positional_encoding(x)
    trad_time = time.time() - start_time

    print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}, dim={dim}")
    print(f"Iterations: {num_iterations}")
    print(f"RoPE average time: {rope_time/num_iterations*1000:.3f}ms")
    print(f"Traditional PE average time: {trad_time/num_iterations*1000:.3f}ms")

    if rope_time < trad_time:
        print(f"RoPE is {trad_time/rope_time:.2f}x faster than traditional PE")
    else:
        print(f"Traditional PE is {rope_time/trad_time:.2f}x faster than RoPE")


def main():
    """Main function"""
    print("RoPE (Rotary Position Embedding) Simple Demo")
    print("=" * 60)

    demo_rope_basics()
    demo_position_preservation()
    compare_with_traditional()
    mathematical_explanation()
    performance_comparison()

    print("\n" + "="*60)
    print("Demo Summary")
    print("="*60)
    print("1. RoPE encodes position through rotation matrices")
    print("2. Preserves relative position information effectively")
    print("3. No additional parameters required")
    print("4. Computationally efficient for long sequences")
    print("5. Widely used in modern transformer models")
    print("\nKey advantages:")
    print("• Better extrapolation to longer sequences")
    print("• More stable training dynamics")
    print("• Natural handling of relative positions")
    print("• Easy to implement and integrate")


if __name__ == "__main__":
    main()