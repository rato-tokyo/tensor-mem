"""Standard Transformer LLM model for comparison.

StandardTransformerLM: Traditional transformer with positional encoding (baseline).

Declarative Configuration:
    model = StandardTransformerLM(
        vocab_size=32000,
        max_len=512,
        layers=[
            StandardTransformerBlock(d_model=256, num_heads=4, d_ff=1024),
            StandardTransformerBlock(d_model=256, num_heads=4, d_ff=1024),
        ],
    )
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .block import StandardTransformerBlock
from .positional_encoding import SinusoidalPositionalEncoding


class StandardTransformerLM(nn.Module):
    """Standard transformer language model with positional encoding.

    This serves as the baseline for comparison with tensor memory LLM.
    Uses sinusoidal positional encoding and standard softmax attention.

    Declarative Configuration:
        model = StandardTransformerLM(
            vocab_size=32000,
            max_len=512,
            layers=[
                StandardTransformerBlock(d_model=256, num_heads=4, d_ff=1024),
                StandardTransformerBlock(d_model=256, num_heads=4, d_ff=1024),
            ],
        )
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        layers: list[StandardTransformerBlock],
    ) -> None:
        """Initialize StandardTransformerLM.

        Args:
            vocab_size: Size of vocabulary.
            max_len: Maximum sequence length for positional encoding.
            layers: List of pre-configured StandardTransformerBlock instances.
        """
        super().__init__()
        if not layers:
            raise ValueError("layers must not be empty")

        # Get d_model from first layer
        self.d_model = layers[0].d_model

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(self.d_model, max_len)

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
        x = self.pos_encoding(x)

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
