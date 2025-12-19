"""Linear Memory Attention implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f

from .memory import MultiHeadMemory


class LinearMemoryAttention(nn.Module):
    """
    Linear Attention with Tensor Product Memory.

    A simple multi-head attention layer that uses ONLY tensor product memory.
    No local attention, no GQA - just straightforward memory-based retrieval.

    Uses Dependency Injection - receives a pre-configured MultiHeadMemory instance.

    Design:
        1. Project input to Q, K, V
        2. (Optional) L2 normalize Q, K, V for numerical stability
        3. Split into heads
        4. RETRIEVE from memory using Q
        5. UPDATE memory with K, V
        6. Merge heads and project output

    Args:
        memory: Pre-configured MultiHeadMemory instance.
        hidden_size: Hidden dimension of the model.
        bias: Whether to use bias in projections.
        normalize_qkv: Whether to L2 normalize Q, K, V after projection.
            Recommended for fp16 training to prevent overflow.

    Example:
        >>> from tensor_mem import TensorMemory, MultiHeadMemory
        >>>
        >>> # 1. Create memory instances
        >>> memories = [TensorMemory(dim=64, eps=1e-6) for _ in range(8)]
        >>>
        >>> # 2. Create MultiHeadMemory
        >>> mh_memory = MultiHeadMemory(memories)
        >>>
        >>> # 3. Inject into LinearMemoryAttention
        >>> attn = LinearMemoryAttention(
        ...     memory=mh_memory,
        ...     hidden_size=512,
        ...     bias=True,
        ...     normalize_qkv=False,
        ... )
        >>>
        >>> x = torch.randn(2, 128, 512)
        >>> output = attn(x)
        >>> attn.reset_memory()  # For new sequence
    """

    def __init__(
        self,
        memory: MultiHeadMemory,
        hidden_size: int,
        bias: bool,
        normalize_qkv: bool,
    ) -> None:
        """Initialize LinearMemoryAttention."""
        super().__init__()

        self.hidden_size = hidden_size
        self.memory = memory
        self.num_heads = memory.num_heads
        self.head_dim = memory.head_dim
        self.normalize_qkv = normalize_qkv

        proj_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.o_proj = nn.Linear(proj_dim, hidden_size, bias=False)

    def reset_memory(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Reset memory for a new sequence."""
        self.memory.reset(device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq, hidden_size].

        Returns:
            Output tensor [batch, seq, hidden_size].
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Initialize memory if needed
        if not self.memory.is_initialized:
            self.reset_memory(device=device, dtype=dtype)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Optional L2 normalization for numerical stability (recommended for fp16)
        if self.normalize_qkv:
            q = f.normalize(q, p=2, dim=-1)
            k = f.normalize(k, p=2, dim=-1)
            v = f.normalize(v, p=2, dim=-1)

        # Split into heads: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 1: RETRIEVE from memory
        output = self.memory.retrieve(q)

        # Step 2: UPDATE memory
        self.memory.update(k, v)

        # Merge heads: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
        output = output.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

        # Project output
        result: torch.Tensor = self.o_proj(output)

        return result

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}"
        )
