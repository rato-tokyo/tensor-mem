"""Standard transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn


class StandardTransformerBlock(nn.Module):
    """Standard transformer block with multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm architecture and causal masking."""
        normed = self.norm1(x)
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        attn_out, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            is_causal=True,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
