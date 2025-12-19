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
