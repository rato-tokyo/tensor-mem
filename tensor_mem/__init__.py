"""
tensor-mem: Linear Attention with Tensor Product Memory.

A lightweight library for building transformers with infinite context using
linear attention and tensor product memory mechanisms.

Features:
    - Linear O(n) complexity instead of O(n²) for attention
    - Infinite context support via persistent tensor memory
    - HuggingFace transformers compatible interface
    - GQA (Grouped Query Attention) support

Example:
    >>> import torch
    >>> from tensor_mem import TensorMemory, LinearMemoryAttention
    >>>
    >>> # Low-level memory API
    >>> memory = TensorMemory(memory_dim=768)
    >>> memory.reset(device="cuda", dtype=torch.float16)
    >>> keys = torch.randn(2, 100, 768, device="cuda", dtype=torch.float16)
    >>> values = torch.randn(2, 100, 768, device="cuda", dtype=torch.float16)
    >>> memory.update(keys, values)
    >>> queries = torch.randn(2, 10, 768, device="cuda", dtype=torch.float16)
    >>> output = memory.retrieve(queries)
    >>>
    >>> # High-level attention API
    >>> attn = LinearMemoryAttention(
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ... )
    >>> hidden = torch.randn(2, 100, 768)
    >>> output, _, _ = attn(hidden)

References:
    - Infini-attention: https://arxiv.org/abs/2404.07143
    - Linear Transformers Are Secretly Fast Weight Programmers: https://arxiv.org/abs/2102.11174
"""

from __future__ import annotations

from .attention import LinearMemoryAttention
from .memory import TensorMemory
from .utils import elu_plus_one, repeat_kv

__version__ = "0.1.0"
__all__ = [
    "TensorMemory",
    "LinearMemoryAttention",
    "elu_plus_one",
    "repeat_kv",
]
