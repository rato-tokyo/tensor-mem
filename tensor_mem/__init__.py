"""tensor-mem: Linear Attention with Tensor Product Memory.

This package provides:
- Memory: TensorMemory, DecayingTensorMemory, MultiHeadMemory
- Attention: LinearMemoryAttention
- LLM: TensorMemoryLM, TensorMemoryBlock, configuration classes
"""

from .attention import LinearMemoryAttention
from .llm import (
    AttentionConfig,
    LMConfig,
    TensorMemoryBlock,
    TensorMemoryLM,
    TransformerBlockConfig,
    create_tensor_memory_lm,
    large_config,
    medium_config,
    small_config,
)
from .memory import (
    BaseTensorMemory,
    DecayingMemoryConfig,
    DecayingTensorMemory,
    MemoryConfig,
    MultiHeadMemory,
    TensorMemory,
    default_decaying_memory_config,
    default_memory_config,
)

__all__ = [
    # Memory classes
    "BaseTensorMemory",
    "TensorMemory",
    "DecayingTensorMemory",
    "MultiHeadMemory",
    # Memory config classes
    "MemoryConfig",
    "DecayingMemoryConfig",
    "default_memory_config",
    "default_decaying_memory_config",
    # Attention
    "LinearMemoryAttention",
    # LLM classes
    "TensorMemoryBlock",
    "TensorMemoryLM",
    # LLM Config classes
    "AttentionConfig",
    "LMConfig",
    "TransformerBlockConfig",
    # Preset configs
    "large_config",
    "medium_config",
    "small_config",
    # Factory functions
    "create_tensor_memory_lm",
]

__version__ = "0.1.0"
