"""Tensor Memory LLM models and configuration.

This module provides:
- TensorMemoryLM: Pure tensor product memory LLM (NoPE)
- TensorMemoryBlock: Transformer block using tensor product memory
- Configuration dataclasses for centralized settings
- Factory functions for creating models
"""

from __future__ import annotations

from tensor_mem.llm.config import (
    AttentionConfig,
    DecayingMemoryConfig,
    LMConfig,
    MemoryConfig,
    TransformerBlockConfig,
    large_config,
    medium_config,
    small_config,
)
from tensor_mem.llm.models import (
    TensorMemoryBlock,
    TensorMemoryLM,
    create_tensor_memory_lm,
)

__all__ = [
    # Config classes
    "AttentionConfig",
    "DecayingMemoryConfig",
    "LMConfig",
    "MemoryConfig",
    "TransformerBlockConfig",
    # Preset configs
    "large_config",
    "medium_config",
    "small_config",
    # Model classes
    "TensorMemoryBlock",
    "TensorMemoryLM",
    # Factory functions
    "create_tensor_memory_lm",
]
