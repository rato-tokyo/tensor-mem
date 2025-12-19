"""Model and experiment configuration.

Declarative Configuration: All settings and model structures are explicitly defined here.
No CLI parameters, no factory functions, no dynamic generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from baseline import StandardTransformerBlock, StandardTransformerLM
from tensor_mem import Layer, MemoryConfig, TensorMemory, TensorMemoryLM


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


EXPERIMENT_CONFIG = ExperimentConfig()


# =============================================================================
# Memory Configuration
# =============================================================================

MEMORY_CONFIG = MemoryConfig(
    dim=EXPERIMENT_CONFIG.head_dim,
    eps=1e-6,
    use_delta_rule=False,
    max_delta=10.0,
    max_memory=100.0,
    max_norm=1000.0,
)


# =============================================================================
# Model Definitions - Declarative Configuration
# =============================================================================

# Note: vocab_size is set later based on actual vocabulary
# These are template functions that create models with explicit structure


def create_tensor_memory_model(vocab_size: int) -> TensorMemoryLM:
    """Create TensorMemoryLM.

    Declarative Configuration: 4 layers, 4 heads each - structure is visible.
    """
    cfg = EXPERIMENT_CONFIG
    return TensorMemoryLM(
        vocab_size=vocab_size,
        layers=[
            Layer(
                [TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG)],
                hidden_size=cfg.d_model,
                d_ff=cfg.d_ff,
                bias=True,
                normalize_qkv=False,
            ),
            Layer(
                [TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG)],
                hidden_size=cfg.d_model,
                d_ff=cfg.d_ff,
                bias=True,
                normalize_qkv=False,
            ),
            Layer(
                [TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG)],
                hidden_size=cfg.d_model,
                d_ff=cfg.d_ff,
                bias=True,
                normalize_qkv=False,
            ),
            Layer(
                [TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG)],
                hidden_size=cfg.d_model,
                d_ff=cfg.d_ff,
                bias=True,
                normalize_qkv=False,
            ),
        ],
    )


def create_standard_transformer_model(vocab_size: int) -> StandardTransformerLM:
    """Create StandardTransformerLM.

    Declarative Configuration: 4 layers - structure is visible.
    """
    cfg = EXPERIMENT_CONFIG
    return StandardTransformerLM(
        vocab_size=vocab_size,
        max_len=cfg.seq_len,
        layers=[
            StandardTransformerBlock(d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff),
            StandardTransformerBlock(d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff),
            StandardTransformerBlock(d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff),
            StandardTransformerBlock(d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff),
        ],
    )
