"""Baseline models for comparing with tensor-mem LLM.

This module provides:
- StandardTransformerLM: Traditional transformer with positional encoding (baseline)
- TensorMemoryLM: Pure tensor product memory LLM (NoPE)
"""

from __future__ import annotations

from baseline.models import StandardTransformerLM, TensorMemoryLM

__all__ = [
    "StandardTransformerLM",
    "TensorMemoryLM",
]
