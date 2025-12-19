"""tensor-mem: Linear Attention with Tensor Product Memory.

This package provides:
- Memory: TensorMemory, DecayingTensorMemory, MultiHeadMemory
- Attention: LinearMemoryAttention
- Layer: FeedForwardLayer, PreNormBlock
- LLM: TensorMemoryLM, Layer

Declarative Configuration:
    model = TensorMemoryLM(
        vocab_size=32000,
        layers=[
            Layer([TensorMemory(config), ...], hidden_size=256, ...),
            Layer([TensorMemory(config), ...], hidden_size=256, ...),
        ],
    )
"""

from .attention import LinearMemoryAttention
from .layer import (
    FeedForwardLayer,
    PreNormBlock,
)
from .llm import (
    Layer,
    TensorMemoryLM,
)
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
    "FeedForwardLayer",
    "PreNormBlock",
    # LLM classes (Declarative Configuration)
    "Layer",
    "TensorMemoryLM",
]

__version__ = "0.1.0"
