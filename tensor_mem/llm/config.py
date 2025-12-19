"""Configuration dataclasses for LLM models.

All configuration is centralized here. No default arguments in classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for TensorMemory / DecayingTensorMemory.

    All fields are required - no defaults.
    """

    dim: int
    eps: float
    use_delta_rule: bool
    max_delta: float
    max_memory: float
    max_norm: float


@dataclass(frozen=True)
class DecayingMemoryConfig(MemoryConfig):
    """Configuration for DecayingTensorMemory.

    Extends MemoryConfig with decay parameter.
    """

    decay: float


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for LinearMemoryAttention.

    All fields are required - no defaults.
    """

    hidden_size: int
    num_heads: int
    head_dim: int
    bias: bool
    normalize_qkv: bool


@dataclass(frozen=True)
class TransformerBlockConfig:
    """Configuration for TensorMemoryBlock.

    All fields are required - no defaults.
    """

    d_ff: int
    dropout: float


@dataclass(frozen=True)
class LMConfig:
    """Configuration for TensorMemoryLM.

    This is the top-level config that contains all sub-configs.
    All fields are required - no defaults.
    """

    # Model architecture
    vocab_size: int
    num_layers: int
    dropout: float

    # Sub-configs
    memory: MemoryConfig | DecayingMemoryConfig
    attention: AttentionConfig
    block: TransformerBlockConfig

    @property
    def d_model(self) -> int:
        """Model dimension (derived from attention config)."""
        return self.attention.hidden_size

    @property
    def num_heads(self) -> int:
        """Number of attention heads (derived from attention config)."""
        return self.attention.num_heads

    @property
    def head_dim(self) -> int:
        """Dimension per head (derived from attention config)."""
        return self.attention.head_dim


# Shared memory configurations (DRY - Don't Repeat Yourself)
_STANDARD_MEMORY_CONFIG = MemoryConfig(
    dim=64,
    eps=1e-6,
    use_delta_rule=False,
    max_delta=10.0,
    max_memory=100.0,
    max_norm=1000.0,
)

_DECAYING_MEMORY_CONFIG = DecayingMemoryConfig(
    dim=64,
    eps=1e-6,
    use_delta_rule=False,
    max_delta=10.0,
    max_memory=100.0,
    max_norm=1000.0,
    decay=0.95,
)


def _get_memory_config(memory_type: Literal["standard", "decaying"]) -> MemoryConfig | DecayingMemoryConfig:
    """Get memory config by type."""
    if memory_type == "standard":
        return _STANDARD_MEMORY_CONFIG
    return _DECAYING_MEMORY_CONFIG


# Preset configurations for common use cases


def small_config(vocab_size: int, memory_type: Literal["standard", "decaying"]) -> LMConfig:
    """Small model config for testing (3M params).

    Args:
        vocab_size: Vocabulary size
        memory_type: "standard" for TensorMemory, "decaying" for DecayingTensorMemory
    """
    return LMConfig(
        vocab_size=vocab_size,
        num_layers=4,
        dropout=0.1,
        memory=_get_memory_config(memory_type),
        attention=AttentionConfig(
            hidden_size=256,
            num_heads=4,
            head_dim=64,
            bias=True,
            normalize_qkv=False,
        ),
        block=TransformerBlockConfig(
            d_ff=1024,
            dropout=0.1,
        ),
    )


def medium_config(vocab_size: int, memory_type: Literal["standard", "decaying"]) -> LMConfig:
    """Medium model config (25M params).

    Args:
        vocab_size: Vocabulary size
        memory_type: "standard" for TensorMemory, "decaying" for DecayingTensorMemory
    """
    return LMConfig(
        vocab_size=vocab_size,
        num_layers=6,
        dropout=0.1,
        memory=_get_memory_config(memory_type),
        attention=AttentionConfig(
            hidden_size=512,
            num_heads=8,
            head_dim=64,
            bias=True,
            normalize_qkv=False,
        ),
        block=TransformerBlockConfig(
            d_ff=2048,
            dropout=0.1,
        ),
    )


def large_config(vocab_size: int, memory_type: Literal["standard", "decaying"]) -> LMConfig:
    """Large model config (110M params).

    Args:
        vocab_size: Vocabulary size
        memory_type: "standard" for TensorMemory, "decaying" for DecayingTensorMemory
    """
    return LMConfig(
        vocab_size=vocab_size,
        num_layers=12,
        dropout=0.1,
        memory=_get_memory_config(memory_type),
        attention=AttentionConfig(
            hidden_size=768,
            num_heads=12,
            head_dim=64,
            bias=True,
            normalize_qkv=False,
        ),
        block=TransformerBlockConfig(
            d_ff=3072,
            dropout=0.1,
        ),
    )
