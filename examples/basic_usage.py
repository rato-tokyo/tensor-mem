"""Basic usage examples for tensor-mem library."""

import torch

from tensor_mem import LinearMemoryAttention, MultiHeadMemory, TensorMemory


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
    dim = 64
    batch_size = 4
    seq_len = 128

    # Initialize memory
    memory = TensorMemory(dim=dim)
    print(f"Created TensorMemory with dim={dim}")
    print(f"  is_initialized: {memory.is_initialized}")

    # Reset memory (initialize buffers)
    memory.reset(device="cpu", dtype=torch.float32)
    print("After reset:")
    print(f"  is_initialized: {memory.is_initialized}")
    print(f"  is_empty: {memory.is_empty}")
    print(f"  M shape: {memory.M.shape}")
    print(f"  z shape: {memory.z.shape}")

    # Update memory with key-value pairs
    keys = torch.randn(batch_size, seq_len, dim)
    values = torch.randn(batch_size, seq_len, dim)
    print(f"\nUpdating memory with keys/values: [{batch_size}, {seq_len}, {dim}]")

    memory.update(keys, values)
    print("After update:")
    print(f"  is_empty: {memory.is_empty}")
    print(f"  z sum: {memory.z.sum().item():.4f}")

    # Retrieve from memory
    query_len = 32
    queries = torch.randn(batch_size, query_len, dim)
    print(f"\nRetrieving with queries: [{batch_size}, {query_len}, {dim}]")

    output = memory.retrieve(queries)
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")

    # Accumulate more information
    print("\n--- Accumulating more sequences ---")
    for i in range(3):
        keys = torch.randn(batch_size, seq_len, dim)
        values = torch.randn(batch_size, seq_len, dim)
        memory.update(keys, values)
        print(f"After update {i + 1}: z sum = {memory.z.sum().item():.4f}")

    print()


def multi_head_memory_example():
    """
    Example: Using MultiHeadMemory for multi-head attention.

    MultiHeadMemory is a wrapper that creates multiple independent
    TensorMemory instances, one per head.
    """
    print("=" * 60)
    print("MultiHeadMemory Example")
    print("=" * 60)

    # Configuration
    num_heads = 8
    head_dim = 64
    batch_size = 2
    seq_len = 128

    # Create multi-head memory
    mh_memory = MultiHeadMemory(num_heads=num_heads, head_dim=head_dim)
    print(f"Created MultiHeadMemory:")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  total memories: {len(mh_memory.memories)}")

    # Reset all memories
    mh_memory.reset(device="cpu", dtype=torch.float32)
    print(f"  is_initialized: {mh_memory.is_initialized}")

    # Update with multi-head tensors: [batch, num_heads, seq, head_dim]
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    print(f"\nUpdating with shape: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    mh_memory.update(keys, values)

    # Retrieve
    query_len = 32
    queries = torch.randn(batch_size, num_heads, query_len, head_dim)
    output = mh_memory.retrieve(queries)
    print(f"Retrieved shape: {output.shape}")

    print()


def linear_memory_attention_example():
    """
    Example: Using LinearMemoryAttention (high-level API).

    LinearMemoryAttention is a drop-in replacement for standard attention
    that uses tensor product memory.
    """
    print("=" * 60)
    print("LinearMemoryAttention Example (High-level API)")
    print("=" * 60)

    # Configuration
    hidden_size = 512
    num_heads = 8
    batch_size = 2
    seq_len = 256

    # Create attention layer
    attn = LinearMemoryAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )
    print("Created LinearMemoryAttention:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {attn.head_dim}")

    # Forward pass
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    print(f"\nInput shape: {hidden_states.shape}")

    output = attn(hidden_states)
    print(f"Output shape: {output.shape}")

    # Process multiple chunks (like in long context)
    print("\n--- Processing multiple chunks ---")
    for chunk_idx in range(3):
        chunk = torch.randn(batch_size, 128, hidden_size)
        output = attn(chunk)
        print(f"Chunk {chunk_idx + 1}: output mean = {output.mean().item():.6f}")

    # Reset for new sequence
    print("\n--- Starting new sequence ---")
    attn.reset_memory()
    print("Memory reset complete")

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
                num_heads=num_heads,
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
            x = x + self.attention(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

        def reset_memory(self):
            self.attention.reset_memory()

    # Create a small model
    hidden_size = 256
    num_heads = 4
    num_layers = 4

    blocks = torch.nn.ModuleList([
        TransformerBlock(hidden_size, num_heads)
        for _ in range(num_layers)
    ])

    print(f"Created {num_layers} transformer blocks")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")

    # Forward pass through all blocks
    x = torch.randn(2, 128, hidden_size)
    print(f"\nInput shape: {x.shape}")

    for i, block in enumerate(blocks):
        x = block(x)
        print(f"After block {i + 1}: mean = {x.mean().item():.6f}")

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
    multi_head_memory_example()
    linear_memory_attention_example()
    custom_transformer_block_example()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
