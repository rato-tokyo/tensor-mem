"""Configuration helpers for training scripts.

These are convenience functions for scripts, not part of the core library.
"""

from __future__ import annotations

from tensor_mem import DecayingMemoryConfig, MemoryConfig


def default_memory_config(dim: int) -> MemoryConfig:
    """Create default memory configuration for training.

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
    """Create default decaying memory configuration for training.

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
