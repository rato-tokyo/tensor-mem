"""Baseline models for comparing with tensor-mem LLM.

This module provides:
- StandardTransformerLM: Traditional transformer with positional encoding (baseline)
- TensorMemoryLM: Pure tensor product memory LLM (NoPE)
- Configuration classes for centralized settings
"""

from __future__ import annotations

from baseline.config import (
    AttentionConfig,
    DecayingMemoryConfig,
    LMConfig,
    MemoryConfig,
    TransformerBlockConfig,
    large_config,
    medium_config,
    small_config,
)
from baseline.models import (
    StandardTransformerLM,
    TensorMemoryBlock,
    TensorMemoryLM,
    create_standard_transformer_lm,
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
    "StandardTransformerLM",
    "TensorMemoryBlock",
    "TensorMemoryLM",
    # Factory functions
    "create_standard_transformer_lm",
    "create_tensor_memory_lm",
]
