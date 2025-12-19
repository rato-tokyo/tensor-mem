"""tensor-mem: Linear Attention with Tensor Product Memory.

This package provides:
- Memory: TensorMemory, DecayingTensorMemory, MultiHeadMemory
- Attention: LinearMemoryAttention
- Layer: TensorMemoryLayer, FeedForwardLayer, PreNormBlock
- LLM: TensorMemoryLM

Declarative Configuration:
    model = TensorMemoryLM(
        vocab_size=32000,
        layers=[
            TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config),
            TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config),
        ],
    )
"""

from .attention import LinearMemoryAttention
from .layer import (
    FeedForwardLayer,
    PreNormBlock,
    TensorMemoryLayer,
)
from .llm import TensorMemoryLM
from .memory import (
    BaseTensorMemory,
    DecayingMemoryConfig,
    DecayingTensorMemory,
    MemoryConfig,
    MultiHeadMemory,
    TensorMemory,
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
    # Attention
    "LinearMemoryAttention",
    # Layer components
    "TensorMemoryLayer",
    "FeedForwardLayer",
    "PreNormBlock",
    # LLM classes (Declarative Configuration)
    "TensorMemoryLM",
]

__version__ = "0.1.0"
