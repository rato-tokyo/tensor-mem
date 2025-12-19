"""tensor-mem: Linear Attention with Tensor Product Memory."""

from .attention import LinearMemoryAttention
from .memory import DecayingTensorMemory, MultiHeadMemory, TensorMemory

__all__ = [
    "TensorMemory",
    "DecayingTensorMemory",
    "MultiHeadMemory",
    "LinearMemoryAttention",
]

__version__ = "0.1.0"
