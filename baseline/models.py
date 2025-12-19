"""Baseline LLM models for comparison.

StandardTransformerLM: Traditional transformer with positional encoding (baseline).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import StandardTransformerConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1), :]
        result: torch.Tensor = self.dropout(x)
        return result


class StandardTransformerBlock(nn.Module):
    """Standard transformer block with multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with pre-norm architecture."""
        normed = self.norm1(x)
        attn_out, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class StandardTransformerLM(nn.Module):
    """Standard transformer language model with positional encoding.

    This serves as the baseline for comparison with tensor memory LLM.
    Uses sinusoidal positional encoding and standard softmax attention.

    Uses config-based initialization - no default arguments.
    """

    def __init__(self, config: StandardTransformerConfig) -> None:
        """Initialize StandardTransformerLM.

        Args:
            config: StandardTransformerConfig containing all model settings.
        """
        super().__init__()
        self.d_model = config.d_model

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.max_len, config.dropout
        )

        self.layers = nn.ModuleList(
            [
                StandardTransformerBlock(
                    config.d_model, config.num_heads, config.d_ff, config.dropout
                )
                for _ in range(config.num_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

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


def create_standard_transformer_lm(config: StandardTransformerConfig) -> StandardTransformerLM:
    """Factory function to create StandardTransformerLM from config.

    Args:
        config: StandardTransformerConfig containing all model settings.

    Returns:
        Configured StandardTransformerLM instance
    """
    return StandardTransformerLM(config)
