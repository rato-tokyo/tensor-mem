"""Basic usage examples for tensor-mem library."""

import torch

from tensor_mem import LinearMemoryAttention, TensorMemory


def tensor_memory_example():
    """
    Example: Using TensorMemory directly (low-level API).

    TensorMemory is the core memory component that stores key-value pairs
    and retrieves values based on queries.
    """
    print("=" * 60)
    print("TensorMemory Example (Low-level API)")
    print("=" * 60)

    # Configuration
    memory_dim = 768
    batch_size = 4
    seq_len = 128

    # Initialize memory
    memory = TensorMemory(memory_dim=memory_dim, eps=1e-6)
    print(f"Created TensorMemory with dim={memory_dim}")
    print(f"  is_initialized: {memory.is_initialized}")

    # Reset memory (initialize buffers)
    memory.reset(device="cpu", dtype=torch.float32)
    print("After reset:")
    print(f"  is_initialized: {memory.is_initialized}")
    print(f"  is_empty: {memory.is_empty}")
    print(f"  M shape: {memory.M.shape}")
    print(f"  z shape: {memory.z.shape}")

    # Simulate processing a sequence
    keys = torch.randn(batch_size, seq_len, memory_dim)
    values = torch.randn(batch_size, seq_len, memory_dim)
    print(f"\nUpdating memory with keys/values: [{batch_size}, {seq_len}, {memory_dim}]")

    memory.update(keys, values)
    print("After update:")
    print(f"  is_empty: {memory.is_empty}")
    print(f"  z sum: {memory.z.sum().item():.4f}")

    # Retrieve from memory
    query_len = 32
    queries = torch.randn(batch_size, query_len, memory_dim)
    print(f"\nRetrieving with queries: [{batch_size}, {query_len}, {memory_dim}]")

    output = memory.retrieve(queries)
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")

    # Accumulate more information
    print("\n--- Accumulating more sequences ---")
    for i in range(3):
        keys = torch.randn(batch_size, seq_len, memory_dim)
        values = torch.randn(batch_size, seq_len, memory_dim)
        memory.update(keys, values)
        print(f"After update {i + 1}: z sum = {memory.z.sum().item():.4f}")

    # Reset for new sequence
    print("\n--- Resetting memory ---")
    memory.reset()
    print(f"After reset: is_empty = {memory.is_empty}")

    print()


def linear_memory_attention_example():
    """
    Example: Using LinearMemoryAttention (high-level API).

    LinearMemoryAttention is a drop-in replacement for standard attention
    that uses linear attention and persistent memory.
    """
    print("=" * 60)
    print("LinearMemoryAttention Example (High-level API)")
    print("=" * 60)

    # Configuration (similar to LLaMA-7B)
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 32  # MHA (could be 8 for GQA)
    batch_size = 2
    seq_len = 512

    # Create attention layer
    attn = LinearMemoryAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        eps=1e-6,
    )
    print("Created LinearMemoryAttention:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  num_key_value_heads: {num_key_value_heads}")
    print(f"  head_dim: {attn.head_dim}")
    print(f"  num_memories: {len(attn.memories)}")

    # Forward pass
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    print(f"\nInput shape: {hidden_states.shape}")

    output, attn_weights, cache = attn(hidden_states)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights: {attn_weights}")  # None for linear attention
    print(f"Cache: {cache}")  # None, memory is used instead

    # Simulate processing multiple chunks (like in long context)
    print("\n--- Processing multiple chunks ---")
    for chunk_idx in range(3):
        chunk = torch.randn(batch_size, 128, hidden_size)
        output, _, _ = attn(chunk)
        print(f"Chunk {chunk_idx + 1}: output mean = {output.mean().item():.6f}")

    # Reset for new sequence
    print("\n--- Starting new sequence ---")
    attn.reset_memory()
    print("Memory reset complete")

    print()


def gqa_example():
    """
    Example: Using Grouped Query Attention (GQA).

    GQA uses fewer key-value heads than query heads to reduce memory usage.
    """
    print("=" * 60)
    print("Grouped Query Attention (GQA) Example")
    print("=" * 60)

    # Configuration with GQA (4x fewer KV heads)
    hidden_size = 1024
    num_attention_heads = 16
    num_key_value_heads = 4  # 4x fewer KV heads
    batch_size = 2
    seq_len = 256

    attn = LinearMemoryAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )

    print("GQA Configuration:")
    print(f"  Query heads: {num_attention_heads}")
    print(f"  KV heads: {num_key_value_heads}")
    print(f"  Repetition factor: {attn.num_key_value_groups}")
    print(f"  Head dim: {attn.head_dim}")

    # Compare parameter counts
    q_params = attn.q_proj.weight.numel()
    kv_params = attn.k_proj.weight.numel() + attn.v_proj.weight.numel()
    print("\nParameter counts:")
    print(f"  Q projection: {q_params:,}")
    print(f"  KV projection: {kv_params:,}")
    print(f"  KV reduction: {q_params / kv_params:.1f}x")

    # Forward pass
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    output, _, _ = attn(hidden_states)
    print(f"\nOutput shape: {output.shape}")

    print()


def custom_transformer_block_example():
    """
    Example: Building a custom transformer block with LinearMemoryAttention.
    """
    print("=" * 60)
    print("Custom Transformer Block Example")
    print("=" * 60)

    class TransformerBlock(torch.nn.Module):
        """Simple transformer block with linear memory attention."""

        def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
            super().__init__()
            self.attention = LinearMemoryAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
            )
            self.norm1 = torch.nn.LayerNorm(hidden_size)
            self.norm2 = torch.nn.LayerNorm(hidden_size)

            mlp_dim = int(hidden_size * mlp_ratio)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, mlp_dim),
                torch.nn.GELU(),
                torch.nn.Linear(mlp_dim, hidden_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Pre-norm architecture
            x = x + self.attention(self.norm1(x))[0]
            x = x + self.mlp(self.norm2(x))
            return x

        def reset_memory(self):
            self.attention.reset_memory()

    # Create a small model
    hidden_size = 512
    num_heads = 8
    num_layers = 4

    blocks = torch.nn.ModuleList(
        [TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
    )

    print(f"Created {num_layers} transformer blocks")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")

    # Forward pass through all blocks
    x = torch.randn(2, 128, hidden_size)
    print(f"\nInput shape: {x.shape}")

    for i, block in enumerate(blocks):
        x = block(x)
        print(f"After block {i + 1}: shape = {x.shape}, mean = {x.mean().item():.6f}")

    # Reset all memories
    print("\n--- Resetting all memories ---")
    for block in blocks:
        block.reset_memory()
    print("All memories reset")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("tensor-mem Library Examples")
    print("=" * 60 + "\n")

    tensor_memory_example()
    linear_memory_attention_example()
    gqa_example()
    custom_transformer_block_example()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
