"""Tensor Memory LLM model.

TensorMemoryLM: Pure tensor product memory without positional encoding (NoPE).

Declarative Configuration:
    model = TensorMemoryLM(
        vocab_size=32000,
        layers=[
            TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config),
            TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config),
        ],
    )
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from tensor_mem.layer import TensorMemoryLayer


class TensorMemoryLM(nn.Module):
    """Tensor product memory language model without positional encoding (NoPE).

    Declarative Configuration: structure is visible at construction site.

    Example:
        >>> config = MemoryConfig(dim=64, eps=1e-6, use_delta_rule=False)
        >>> model = TensorMemoryLM(
        ...     vocab_size=32000,
        ...     layers=[
        ...         TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config),
        ...         TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config),
        ...     ],
        ... )
    """

    def __init__(self, vocab_size: int, layers: list[TensorMemoryLayer]) -> None:
        super().__init__()

        if not layers:
            raise ValueError("layers list cannot be empty")

        self.d_model = layers[0].attention.hidden_size

        self.embedding = nn.Embedding(vocab_size, self.d_model)

        self.layers = nn.ModuleList(layers)

        self.norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Input token IDs of shape [batch, seq].

        Returns:
            Logits of shape [batch, seq, vocab_size].
        """
        hidden = self._compute_hidden(input_ids)
        logits: torch.Tensor = self.lm_head(hidden)
        return logits

    def forward_with_hidden(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and hidden states.

        Args:
            input_ids: Input token IDs of shape [batch, seq].

        Returns:
            Tuple of (logits, hidden_states):
                - logits: [batch, seq, vocab_size]
                - hidden_states: [batch, seq, d_model]
        """
        hidden = self._compute_hidden(input_ids)
        logits = self.lm_head(hidden)
        return logits, hidden

    def _compute_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute hidden states from input IDs.

        Args:
            input_ids: Input token IDs of shape [batch, seq].

        Returns:
            Hidden states of shape [batch, seq, d_model].
        """
        x = self.embedding(input_ids) * math.sqrt(self.d_model)

        for layer in self.layers:
            x = layer(x)

        out: torch.Tensor = self.norm(x)
        return out

    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states without computing logits.

        Args:
            input_ids: Input token IDs of shape [batch, seq].

        Returns:
            Hidden states of shape [batch, seq, d_model].
        """
        return self._compute_hidden(input_ids)

    def reset_memory(self) -> None:
        """Reset memory state in all layers."""
        for layer in self.layers:
            layer.reset_memory()
