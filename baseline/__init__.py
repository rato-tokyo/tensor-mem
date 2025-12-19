"""Baseline models for comparing with tensor-mem LLM.

This module provides:
- StandardTransformerLM: Traditional transformer with positional encoding (baseline)
"""

from __future__ import annotations

from baseline.models import (
    StandardTransformerLM,
    create_standard_transformer_lm,
)

__all__ = [
    # Model classes
    "StandardTransformerLM",
    # Factory functions
    "create_standard_transformer_lm",
]
