"""Feed-forward layer implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardLayer(nn.Module):
    """Standard feed-forward network with GELU activation.

    Architecture: Linear -> GELU -> Linear

    Args:
        hidden_size: Input and output dimension.
        d_ff: Inner feed-forward dimension.

    Example:
        >>> ffn = FeedForwardLayer(hidden_size=256, d_ff=1024)
        >>> x = torch.randn(2, 10, 256)
        >>> out = ffn(x)  # [2, 10, 256]
    """

    def __init__(
        self,
        hidden_size: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN.

        Args:
            x: Input tensor of shape [batch, seq, hidden_size].

        Returns:
            Output tensor of shape [batch, seq, hidden_size].
        """
        return self.linear2(self.activation(self.linear1(x)))
