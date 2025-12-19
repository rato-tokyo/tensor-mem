"""tensor-mem: Linear Attention with Tensor Product Memory.

This package provides:
- Memory: TensorMemory, DecayingTensorMemory, MultiHeadMemory
- Attention: LinearMemoryAttention
- LLM: TensorMemoryLM, TensorMemoryBlock, configuration classes
"""

from .attention import LinearMemoryAttention
from .llm import (
    AttentionConfig,
    DecayingMemoryConfig,
    LMConfig,
    MemoryConfig,
    TensorMemoryBlock,
    TensorMemoryLM,
    TransformerBlockConfig,
    create_tensor_memory_lm,
    large_config,
    medium_config,
    small_config,
)
from .memory import BaseTensorMemory, DecayingTensorMemory, MultiHeadMemory, TensorMemory

__all__ = [
    # Memory classes
    "BaseTensorMemory",
    "TensorMemory",
    "DecayingTensorMemory",
    "MultiHeadMemory",
    # Attention
    "LinearMemoryAttention",
    # LLM classes
    "TensorMemoryBlock",
    "TensorMemoryLM",
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
    # Factory functions
    "create_tensor_memory_lm",
]

__version__ = "0.1.0"
