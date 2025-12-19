"""Multi-head memory wrapper."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .base import BaseTensorMemory


class MultiHeadMemory(nn.Module):
    """
    Multi-head wrapper for TensorMemory.

    Wraps multiple independent memory instances, one per head.
    Uses Dependency Injection - receives pre-configured memory instances.

    Args:
        memories: List of pre-configured BaseTensorMemory instances.
            All memories should have the same dimension.

    Example:
        >>> from tensor_mem import TensorMemory, DecayingTensorMemory
        >>>
        >>> # Create memory instances with desired configuration
        >>> memories = [
        ...     DecayingTensorMemory(dim=64, decay=0.95, eps=1e-6)
        ...     for _ in range(8)
        ... ]
        >>>
        >>> # Inject into MultiHeadMemory
        >>> mh_memory = MultiHeadMemory(memories)
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

    def __init__(self, memories: Sequence[BaseTensorMemory]) -> None:
        """Initialize MultiHeadMemory with pre-configured memory instances."""
        super().__init__()

        if not memories:
            raise ValueError("memories list cannot be empty")

        self.memories = nn.ModuleList(memories)
        self._num_heads = len(memories)
        self._head_dim = memories[0].dim

    @property
    def num_heads(self) -> int:
        """Number of memory heads."""
        return self._num_heads

    @property
    def head_dim(self) -> int:
        """Dimension per head."""
        return self._head_dim

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
        memory_type = type(self.memories[0]).__name__
        return f"num_heads={self._num_heads}, head_dim={self._head_dim}, memory_type={memory_type}"
