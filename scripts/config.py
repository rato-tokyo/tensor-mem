"""Configuration for training scripts.

All experiment settings are defined here.
CLI parameters are prohibited - use this config file.
"""

from __future__ import annotations

from dataclasses import dataclass

from tensor_mem import DecayingMemoryConfig, MemoryConfig


# =============================================================================
# Memory Configurations
# =============================================================================


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


# =============================================================================
# Experiment Configuration
# =============================================================================


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for WikiText-2 comparison experiment.

    All settings are defined here. No CLI parameters.
    Edit this dataclass to change experiment settings.
    """

    # Device
    device: str = "cuda"

    # Model architecture
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 1024
    vocab_size: int = 10000

    # Training
    max_epochs: int = 50
    patience: int = 2
    seq_len: int = 64
    batch_size: int = 32
    lr: float = 1e-3
    clip: float = 0.5

    # Data
    data_fraction: float = 1.0

    @property
    def head_dim(self) -> int:
        """Compute head dimension from model dimension and number of heads."""
        return self.d_model // self.num_heads


# Default experiment configuration
EXPERIMENT_CONFIG = ExperimentConfig()
