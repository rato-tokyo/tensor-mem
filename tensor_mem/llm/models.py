"""Tensor Memory LLM models.

TensorMemoryLM: Pure tensor product memory without positional encoding (NoPE).
TensorMemoryBlock: Transformer block using tensor product memory attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tensor_mem.attention import LinearMemoryAttention


class TensorMemoryBlock(nn.Module):
    """Transformer block using tensor product memory attention.

    Uses Dependency Injection - receives a pre-configured LinearMemoryAttention.
    """

    def __init__(self, attention: LinearMemoryAttention, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = attention
        d_model = attention.hidden_size

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
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

    Uses Dependency Injection - receives pre-configured layer blocks.
    """

    def __init__(self, vocab_size: int, layers: list[TensorMemoryBlock], dropout: float) -> None:
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
