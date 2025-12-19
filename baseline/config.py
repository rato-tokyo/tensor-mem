"""Configuration dataclasses for baseline models.

All configuration is centralized here. No default arguments in classes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandardTransformerConfig:
    """Configuration for StandardTransformerLM.

    All fields are required - no defaults.
    """

    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    max_len: int
    dropout: float
