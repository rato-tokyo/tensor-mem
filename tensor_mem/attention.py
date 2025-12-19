"""Linear Memory Attention implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .memory import MultiHeadMemory


class LinearMemoryAttention(nn.Module):
    """
    Linear Attention with Tensor Product Memory.

    A simple multi-head attention layer that uses ONLY tensor product memory.
    No local attention, no GQA - just straightforward memory-based retrieval.

    Design:
        1. Project input to Q, K, V
        2. Split into heads
        3. RETRIEVE from memory using Q
        4. UPDATE memory with K, V
        5. Merge heads and project output

    Args:
        hidden_size: Hidden dimension of the model.
        num_heads: Number of attention heads.
        head_dim: Dimension per head. If None, computed as hidden_size // num_heads.
        eps: Small constant for numerical stability.
        bias: Whether to use bias in projections.

    Example:
        >>> attn = LinearMemoryAttention(hidden_size=512, num_heads=8)
        >>> x = torch.randn(2, 128, 512)
        >>> output = attn(x)
        >>> attn.reset_memory()  # For new sequence
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        eps: float = 1e-6,
        bias: bool = True,
    ) -> None:
        """Initialize LinearMemoryAttention."""
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.eps = eps

        proj_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.o_proj = nn.Linear(proj_dim, hidden_size, bias=False)

        self.memory = MultiHeadMemory(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            eps=eps,
        )

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
