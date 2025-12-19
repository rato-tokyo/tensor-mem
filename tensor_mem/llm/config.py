"""Factory functions for creating TensorMemoryLM models.

Declarative Configuration: Model structure is built explicitly.
No hidden num_layers or num_heads - actual objects are created.
"""

from __future__ import annotations

from typing import Literal

from tensor_mem.attention import LinearMemoryAttention
from tensor_mem.memory import (
    DecayingTensorMemory,
    MultiHeadMemory,
    TensorMemory,
    default_decaying_memory_config,
    default_memory_config,
)
from tensor_mem.memory.config import DecayingMemoryConfig, MemoryConfig

from .models import TensorMemoryBlock, TensorMemoryLM


def _create_layer(
    memory_config: MemoryConfig | DecayingMemoryConfig,
    num_heads: int,
    hidden_size: int,
    d_ff: int,
    dropout: float,
    bias: bool,
    normalize_qkv: bool,
) -> TensorMemoryBlock:
    """Create a single TensorMemoryBlock with explicit structure.

    Args:
        memory_config: Configuration for each memory head.
        num_heads: Number of attention heads.
        hidden_size: Hidden dimension.
        d_ff: Feed-forward dimension.
        dropout: Dropout rate.
        bias: Whether to use bias in projections.
        normalize_qkv: Whether to L2 normalize Q, K, V.

    Returns:
        Configured TensorMemoryBlock.
    """
    if isinstance(memory_config, DecayingMemoryConfig):
        memories: list[TensorMemory | DecayingTensorMemory] = [
            DecayingTensorMemory(memory_config) for _ in range(num_heads)
        ]
    else:
        memories = [TensorMemory(memory_config) for _ in range(num_heads)]

    return TensorMemoryBlock(
        attention=LinearMemoryAttention(
            memory=MultiHeadMemory(memories),
            hidden_size=hidden_size,
            bias=bias,
            normalize_qkv=normalize_qkv,
        ),
        d_ff=d_ff,
        dropout=dropout,
    )


def small_model(
    vocab_size: int,
    memory_type: Literal["standard", "decaying"],
) -> TensorMemoryLM:
    """Create a small model (3M params) with 4 layers, 4 heads.

    Args:
        vocab_size: Vocabulary size.
        memory_type: "standard" for TensorMemory, "decaying" for DecayingTensorMemory.

    Returns:
        TensorMemoryLM with structure:
        - 4 layers
        - 4 heads per layer (head_dim=64)
        - hidden_size=256, d_ff=1024
    """
    head_dim = 64
    if memory_type == "standard":
        memory_config: MemoryConfig | DecayingMemoryConfig = default_memory_config(dim=head_dim)
    else:
        memory_config = default_decaying_memory_config(dim=head_dim, decay=0.95)

    return TensorMemoryLM(
        vocab_size=vocab_size,
        dropout=0.1,
        layers=[
            _create_layer(
                memory_config=memory_config,
                num_heads=4,
                hidden_size=256,
                d_ff=1024,
                dropout=0.1,
                bias=True,
                normalize_qkv=False,
            )
            for _ in range(4)  # 4 layers
        ],
    )


def medium_model(
    vocab_size: int,
    memory_type: Literal["standard", "decaying"],
) -> TensorMemoryLM:
    """Create a medium model (25M params) with 6 layers, 8 heads.

    Args:
        vocab_size: Vocabulary size.
        memory_type: "standard" for TensorMemory, "decaying" for DecayingTensorMemory.

    Returns:
        TensorMemoryLM with structure:
        - 6 layers
        - 8 heads per layer (head_dim=64)
        - hidden_size=512, d_ff=2048
    """
    head_dim = 64
    if memory_type == "standard":
        memory_config: MemoryConfig | DecayingMemoryConfig = default_memory_config(dim=head_dim)
    else:
        memory_config = default_decaying_memory_config(dim=head_dim, decay=0.95)

    return TensorMemoryLM(
        vocab_size=vocab_size,
        dropout=0.1,
        layers=[
            _create_layer(
                memory_config=memory_config,
                num_heads=8,
                hidden_size=512,
                d_ff=2048,
                dropout=0.1,
                bias=True,
                normalize_qkv=False,
            )
            for _ in range(6)  # 6 layers
        ],
    )


def large_model(
    vocab_size: int,
    memory_type: Literal["standard", "decaying"],
) -> TensorMemoryLM:
    """Create a large model (110M params) with 12 layers, 12 heads.

    Args:
        vocab_size: Vocabulary size.
        memory_type: "standard" for TensorMemory, "decaying" for DecayingTensorMemory.

    Returns:
        TensorMemoryLM with structure:
        - 12 layers
        - 12 heads per layer (head_dim=64)
        - hidden_size=768, d_ff=3072
    """
    head_dim = 64
    if memory_type == "standard":
        memory_config: MemoryConfig | DecayingMemoryConfig = default_memory_config(dim=head_dim)
    else:
        memory_config = default_decaying_memory_config(dim=head_dim, decay=0.95)

    return TensorMemoryLM(
        vocab_size=vocab_size,
        dropout=0.1,
        layers=[
            _create_layer(
                memory_config=memory_config,
                num_heads=12,
                hidden_size=768,
                d_ff=3072,
                dropout=0.1,
                bias=True,
                normalize_qkv=False,
            )
            for _ in range(12)  # 12 layers
        ],
    )
