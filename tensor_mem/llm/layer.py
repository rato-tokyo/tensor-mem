"""Transformer layer with tensor product memory attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from tensor_mem.attention import LinearMemoryAttention
from tensor_mem.memory import BaseTensorMemory, MultiHeadMemory


class Layer(nn.Module):
    """Single transformer layer with tensor product memory attention.

    Declarative Configuration: receives list of memory instances directly.

    Args:
        memories: List of TensorMemory or DecayingTensorMemory instances.
        hidden_size: Hidden dimension of the model.
        d_ff: Feed-forward dimension.
        bias: Whether to use bias in attention projections.
        normalize_qkv: Whether to L2 normalize Q, K, V.
    """

    def __init__(
        self,
        memories: list[BaseTensorMemory],
        hidden_size: int,
        d_ff: int,
        bias: bool,
        normalize_qkv: bool,
    ) -> None:
        super().__init__()

        self.attention = LinearMemoryAttention(
            memory=MultiHeadMemory(memories),
            hidden_size=hidden_size,
            bias=bias,
            normalize_qkv=normalize_qkv,
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm architecture."""
        normed = self.norm1(x)
        attn_out = self.attention(normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

    def reset_memory(self) -> None:
        """Reset the memory state."""
        self.attention.reset_memory()
