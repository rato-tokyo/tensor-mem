"""Tensor Memory LLM models.

This module provides:
- TensorMemoryLM: Pure tensor product memory LLM (NoPE)
- TensorMemoryBlock: Transformer block using tensor product memory
- Factory functions for creating preset model configurations
"""

from __future__ import annotations

from tensor_mem.llm.config import (
    large_model,
    medium_model,
    small_model,
)
from tensor_mem.llm.models import (
    TensorMemoryBlock,
    TensorMemoryLM,
)

__all__ = [
    # Model classes
    "TensorMemoryBlock",
    "TensorMemoryLM",
    # Factory functions
    "large_model",
    "medium_model",
    "small_model",
]
