"""TensorMemory layer implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from tensor_mem.attention import LinearMemoryAttention
from tensor_mem.layer.ffn import FeedForwardLayer
from tensor_mem.memory import MemoryConfig, MultiHeadMemory, TensorMemory


class TensorMemoryLayer(nn.Module):
    """Transformer layer with tensor product memory attention.

    Pre-norm architecture:
        x = x + attention(LayerNorm(x))
        x = x + ffn(LayerNorm(x))

    Fixed 4 heads configuration.

    Args:
        hidden_size: Hidden dimension of the model (must be divisible by 4).
        d_ff: Feed-forward dimension.
        memory_config: Configuration for TensorMemory instances.

    Example:
        >>> from tensor_mem import MemoryConfig
        >>> from tensor_mem.layer import TensorMemoryLayer
        >>>
        >>> config = MemoryConfig(dim=64, eps=1e-6, use_delta_rule=False)
        >>> layer = TensorMemoryLayer(
        ...     hidden_size=256,
        ...     d_ff=1024,
        ...     memory_config=config,
        ... )
    """

    def __init__(
        self,
        hidden_size: int,
        d_ff: int,
        memory_config: MemoryConfig,
    ) -> None:
        super().__init__()

        # Fixed 4 heads
        memories = [
            TensorMemory(memory_config),
            TensorMemory(memory_config),
            TensorMemory(memory_config),
            TensorMemory(memory_config),
        ]

        self.attention = LinearMemoryAttention(
            memory=MultiHeadMemory(memories),
            hidden_size=hidden_size,
            bias=True,
            normalize_qkv=False,
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardLayer(hidden_size=hidden_size, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm architecture.

        Args:
            x: Input tensor of shape [batch, seq, hidden_size].

        Returns:
            Output tensor of shape [batch, seq, hidden_size].
        """
        normed = self.norm1(x)
        attn_out = self.attention(normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

    def reset_memory(self) -> None:
        """Reset the memory state."""
        self.attention.reset_memory()
