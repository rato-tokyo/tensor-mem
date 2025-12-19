"""Pre-normalization block implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class PreNormBlock(nn.Module):
    """Pre-normalization wrapper for any sublayer.

    Applies LayerNorm before the sublayer with residual connection:
        output = x + sublayer(LayerNorm(x))

    Args:
        hidden_size: Dimension for LayerNorm.
        sublayer: The sublayer to wrap (e.g., attention, FFN).

    Example:
        >>> attention = SomeAttention(hidden_size=256)
        >>> block = PreNormBlock(hidden_size=256, sublayer=attention)
        >>> x = torch.randn(2, 10, 256)
        >>> out = block(x)  # Pre-norm + residual applied
    """

    def __init__(
        self,
        hidden_size: int,
        sublayer: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm and residual.

        Args:
            x: Input tensor of shape [batch, seq, hidden_size].

        Returns:
            Output tensor of shape [batch, seq, hidden_size].
        """
        return x + self.sublayer(self.norm(x))
