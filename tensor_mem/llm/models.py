"""Tensor Memory LLM models.

TensorMemoryLM: Pure tensor product memory without positional encoding (NoPE).
Layer: Single transformer layer with tensor product memory attention.

Declarative Configuration:
    model = TensorMemoryLM(
        vocab_size=32000,
        layers=[
            Layer([TensorMemory(config), TensorMemory(config), ...]),
            Layer([TensorMemory(config), TensorMemory(config), ...]),
        ],
    )
"""

from __future__ import annotations

import math

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
        dropout: Dropout rate.
        bias: Whether to use bias in attention projections.
        normalize_qkv: Whether to L2 normalize Q, K, V.
    """

    def __init__(
        self,
        memories: list[BaseTensorMemory],
        hidden_size: int,
        d_ff: int,
        dropout: float,
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
            nn.Dropout(dropout),
            nn.Linear(d_ff, hidden_size),
            nn.Dropout(dropout),
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


class TensorMemoryLM(nn.Module):
    """Tensor product memory language model without positional encoding (NoPE).

    Declarative Configuration: structure is visible at construction site.

    Example:
        >>> config = default_memory_config(dim=64)
        >>> model = TensorMemoryLM(
        ...     vocab_size=32000,
        ...     dropout=0.1,
        ...     layers=[
        ...         Layer([TensorMemory(config), TensorMemory(config)], hidden_size=128, ...),
        ...         Layer([TensorMemory(config), TensorMemory(config)], hidden_size=128, ...),
        ...     ],
        ... )
    """

    def __init__(self, vocab_size: int, dropout: float, layers: list[Layer]) -> None:
        super().__init__()

        if not layers:
            raise ValueError("layers list cannot be empty")

        self.d_model = layers[0].attention.hidden_size

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(layers)

        self.norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x)

        hidden = self.norm(x)
        logits = self.lm_head(hidden)

        if return_hidden:
            return logits, hidden
        result: torch.Tensor = logits
        return result

    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states without computing logits."""
        _, hidden = self.forward(input_ids, return_hidden=True)
        return hidden

    def reset_memory(self) -> None:
        """Reset memory state in all layers."""
        for layer in self.layers:
            layer.reset_memory()
