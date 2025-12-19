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
    """

    dim: int
    eps: float
    use_delta_rule: bool


@dataclass(frozen=True)
class DecayingMemoryConfig(MemoryConfig):
    """Configuration for DecayingTensorMemory.

    Extends MemoryConfig with decay parameter.
    All fields are required - no defaults.

    Args:
        dim: Dimension of the memory vectors.
        eps: Small constant for numerical stability.
        use_delta_rule: Whether to use Delta Rule for updates.
        decay: Decay factor in range (0, 1). Higher = longer memory.
    """

    decay: float
