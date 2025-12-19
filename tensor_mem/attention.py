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

    Design:
        1. Project input to Q, K, V
        2. (Optional) L2 normalize Q, K, V for numerical stability
        3. Split into heads
        4. RETRIEVE from memory using Q
        5. UPDATE memory with K, V
        6. Merge heads and project output

    Args:
        hidden_size: Hidden dimension of the model.
        num_heads: Number of attention heads.
        head_dim: Dimension per head. If None, computed as hidden_size // num_heads.
        eps: Small constant for numerical stability.
        bias: Whether to use bias in projections.
        use_delta_rule: Whether to use Delta Rule for memory updates.
        normalize_qkv: Whether to L2 normalize Q, K, V after projection.
            Recommended for fp16 training to prevent overflow.

    Raises:
        ValueError: If head_dim is None and hidden_size is not divisible by num_heads.

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
        use_delta_rule: bool = False,
        normalize_qkv: bool = False,
    ) -> None:
        """Initialize LinearMemoryAttention."""
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.normalize_qkv = normalize_qkv

        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by "
                    f"num_heads ({num_heads}) when head_dim is not specified"
                )
            self.head_dim = hidden_size // num_heads
        else:
            self.head_dim = head_dim

        proj_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.o_proj = nn.Linear(proj_dim, hidden_size, bias=False)

        self.memory = MultiHeadMemory(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            eps=eps,
            use_delta_rule=use_delta_rule,
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
