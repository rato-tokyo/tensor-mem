"""Multi-head memory wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import TensorMemory


class MultiHeadMemory(nn.Module):
    """
    Multi-head wrapper for TensorMemory.

    Creates multiple independent TensorMemory instances, one per head.
    This is a convenience wrapper for multi-head attention patterns.

    Args:
        num_heads: Number of memory heads.
        head_dim: Dimension per head.
        eps: Small constant for numerical stability.
        use_delta_rule: Whether to use Delta Rule for updates.

    Example:
        >>> mh_memory = MultiHeadMemory(num_heads=8, head_dim=64)
        >>> mh_memory.reset(device="cuda")
        >>>
        >>> # keys/values: [batch, num_heads, seq, head_dim]
        >>> keys = torch.randn(4, 8, 128, 64, device="cuda")
        >>> values = torch.randn(4, 8, 128, 64, device="cuda")
        >>> mh_memory.update(keys, values)
        >>>
        >>> queries = torch.randn(4, 8, 32, 64, device="cuda")
        >>> output = mh_memory.retrieve(queries)  # [4, 8, 32, 64]
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        use_delta_rule: bool = False,
    ) -> None:
        """Initialize MultiHeadMemory."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.memories = nn.ModuleList(
            [
                TensorMemory(dim=head_dim, eps=eps, use_delta_rule=use_delta_rule)
                for _ in range(num_heads)
            ]
        )

    @property
    def is_initialized(self) -> bool:
        """Check if all memories are initialized."""
        return all(m.is_initialized for m in self.memories)

    def reset(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Reset all memories."""
        for memory in self.memories:
            memory.reset(device=device, dtype=dtype)

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Update all memories.

        Args:
            keys: [batch, num_heads, seq, head_dim]
            values: [batch, num_heads, seq, head_dim]
        """
        for h, memory in enumerate(self.memories):
            memory.update(keys[:, h], values[:, h])

    def retrieve(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from all memories.

        Args:
            queries: [batch, num_heads, seq, head_dim]

        Returns:
            [batch, num_heads, seq, head_dim]
        """
        outputs = []
        for h, memory in enumerate(self.memories):
            out = memory.retrieve(queries[:, h])
            outputs.append(out)

        return torch.stack(outputs, dim=1)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}"
