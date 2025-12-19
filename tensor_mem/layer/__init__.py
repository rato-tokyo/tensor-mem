"""Layer components for tensor-mem.

This module provides reusable layer components:
- FeedForwardLayer: Standard FFN with GELU activation
- PreNormBlock: Pre-normalization wrapper for any sublayer
"""

from .ffn import FeedForwardLayer
from .prenorm import PreNormBlock

__all__ = [
    "FeedForwardLayer",
    "PreNormBlock",
]
