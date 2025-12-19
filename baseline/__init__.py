"""Baseline models for comparing with tensor-mem LLM.

This module provides:
- StandardTransformerLM: Traditional transformer with positional encoding (baseline)
- StandardTransformerBlock: Single transformer block

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

from baseline.block import StandardTransformerBlock
from baseline.models import StandardTransformerLM
from baseline.positional_encoding import SinusoidalPositionalEncoding

__all__ = [
    "SinusoidalPositionalEncoding",
    "StandardTransformerBlock",
    "StandardTransformerLM",
]
