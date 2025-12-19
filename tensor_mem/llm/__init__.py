"""Tensor Memory LLM models.

This module provides:
- TensorMemoryLM: Pure tensor product memory LLM (NoPE)
- Layer: Single transformer layer with tensor product memory

Declarative Configuration:
    model = TensorMemoryLM(
        vocab_size=32000,
        layers=[
            Layer([TensorMemory(config), ...], hidden_size=256, ...),
            Layer([TensorMemory(config), ...], hidden_size=256, ...),
        ],
    )
"""

from __future__ import annotations

from tensor_mem.llm.layer import Layer
from tensor_mem.llm.models import TensorMemoryLM

__all__ = [
    "Layer",
    "TensorMemoryLM",
]
