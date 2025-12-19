"""Layer components for tensor-mem.

This module provides reusable layer components:
- TensorMemoryLayer: Transformer layer with tensor product memory attention
- FeedForwardLayer: Standard FFN with GELU activation
- PreNormBlock: Pre-normalization wrapper for any sublayer
"""

from .ffn import FeedForwardLayer
from .prenorm import PreNormBlock
from .tensor_memory_layer import TensorMemoryLayer

__all__ = [
    "TensorMemoryLayer",
    "FeedForwardLayer",
    "PreNormBlock",
]
