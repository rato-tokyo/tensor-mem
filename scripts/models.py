"""Model definitions for experiments.

Declarative Configuration: All model structures are explicitly defined here.
No factory functions, no dynamic generation.
"""

from __future__ import annotations

import torch.nn as nn

from baseline import StandardTransformerBlock, StandardTransformerLM
from config import EXPERIMENT_CONFIG, default_memory_config
from tensor_mem import Layer, TensorMemory, TensorMemoryLM

# =============================================================================
# Configuration
# =============================================================================

cfg = EXPERIMENT_CONFIG
memory_config = default_memory_config(dim=cfg.head_dim)


# =============================================================================
# Model Definitions - Declarative Configuration
# =============================================================================


def create_tensor_memory_model(vocab_size: int) -> TensorMemoryLM:
    """Create TensorMemoryLM with Declarative Configuration.

    Structure: 4 layers, 4 heads each - visible and explicit.
    """
    return TensorMemoryLM(
        vocab_size=vocab_size,
        layers=[
            Layer(
                [TensorMemory(memory_config), TensorMemory(memory_config), TensorMemory(memory_config), TensorMemory(memory_config)],
                hidden_size=cfg.d_model,
                d_ff=cfg.d_ff,
                bias=True,
                normalize_qkv=False,
            ),
            Layer(
                [TensorMemory(memory_config), TensorMemory(memory_config), TensorMemory(memory_config), TensorMemory(memory_config)],
                hidden_size=cfg.d_model,
                d_ff=cfg.d_ff,
                bias=True,
                normalize_qkv=False,
            ),
            Layer(
                [TensorMemory(memory_config), TensorMemory(memory_config), TensorMemory(memory_config), TensorMemory(memory_config)],
                hidden_size=cfg.d_model,
                d_ff=cfg.d_ff,
                bias=True,
                normalize_qkv=False,
            ),
            Layer(
                [TensorMemory(memory_config), TensorMemory(memory_config), TensorMemory(memory_config), TensorMemory(memory_config)],
                hidden_size=cfg.d_model,
                d_ff=cfg.d_ff,
                bias=True,
                normalize_qkv=False,
            ),
        ],
    )


def create_standard_transformer_model(vocab_size: int) -> StandardTransformerLM:
    """Create StandardTransformerLM with Declarative Configuration.

    Structure: 4 layers - visible and explicit.
    """
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
