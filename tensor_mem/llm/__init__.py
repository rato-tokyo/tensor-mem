"""Tensor Memory LLM models.

This module provides:
- TensorMemoryLM: Pure tensor product memory LLM (NoPE)

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

from tensor_mem.llm.models import TensorMemoryLM

__all__ = [
    "TensorMemoryLM",
]
