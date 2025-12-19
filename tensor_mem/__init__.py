"""tensor-mem: Linear Attention with Tensor Product Memory.

This package provides:
- Memory: TensorMemory, DecayingTensorMemory, MultiHeadMemory
- Attention: LinearMemoryAttention
- LLM: TensorMemoryLM, TensorMemoryBlock, factory functions
"""

from .attention import LinearMemoryAttention
from .llm import (
    TensorMemoryBlock,
    TensorMemoryLM,
    large_model,
    medium_model,
    small_model,
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
    # Factory functions (Declarative Configuration)
    "large_model",
    "medium_model",
    "small_model",
]

__version__ = "0.1.0"
