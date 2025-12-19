"""Baseline models for comparing with tensor-mem LLM.

This module provides:
- StandardTransformerLM: Traditional transformer with positional encoding (baseline)
- StandardTransformerBlock: Single transformer block

Declarative Configuration:
    model = StandardTransformerLM(
        vocab_size=32000,
        max_len=512,
        dropout=0.1,
        layers=[
            StandardTransformerBlock(d_model=256, num_heads=4, d_ff=1024, dropout=0.1),
            StandardTransformerBlock(d_model=256, num_heads=4, d_ff=1024, dropout=0.1),
        ],
    )
"""

from __future__ import annotations

from baseline.models import (
    StandardTransformerBlock,
    StandardTransformerLM,
)

__all__ = [
    "StandardTransformerBlock",
    "StandardTransformerLM",
]
