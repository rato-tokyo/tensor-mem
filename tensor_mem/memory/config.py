"""Configuration dataclasses for Memory classes.

All configuration is centralized here. No default arguments in classes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for TensorMemory.

    All fields are required - no defaults.

    Args:
        dim: Dimension of the memory vectors.
        eps: Small constant for numerical stability.
        use_delta_rule: Whether to use Delta Rule for updates.
        max_delta: Maximum absolute value for update deltas.
        max_memory: Maximum absolute value for memory matrix M.
        max_norm: Maximum value for normalization term z.
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
    All fields are required - no defaults.

    Args:
        dim: Dimension of the memory vectors.
        eps: Small constant for numerical stability.
        use_delta_rule: Whether to use Delta Rule for updates.
        max_delta: Maximum absolute value for update deltas.
        max_memory: Maximum absolute value for memory matrix M.
        max_norm: Maximum value for normalization term z.
        decay: Decay factor in range (0, 1). Higher = longer memory.
    """

    decay: float


def default_memory_config(dim: int) -> MemoryConfig:
    """Create default memory configuration.

    Args:
        dim: Dimension of the memory vectors.

    Returns:
        MemoryConfig with standard default values.
    """
    return MemoryConfig(
        dim=dim,
        eps=1e-6,
        use_delta_rule=False,
        max_delta=10.0,
        max_memory=100.0,
        max_norm=1000.0,
    )


def default_decaying_memory_config(dim: int, decay: float) -> DecayingMemoryConfig:
    """Create default decaying memory configuration.

    Args:
        dim: Dimension of the memory vectors.
        decay: Decay factor in range (0, 1).

    Returns:
        DecayingMemoryConfig with standard default values.
    """
    return DecayingMemoryConfig(
        dim=dim,
        eps=1e-6,
        use_delta_rule=False,
        max_delta=10.0,
        max_memory=100.0,
        max_norm=1000.0,
        decay=decay,
    )
