"""Model and experiment configuration.

Declarative Configuration: All settings and model structures are explicitly defined here.
No CLI parameters, no factory functions, no dynamic generation.
"""

from __future__ import annotations

from baseline import StandardTransformerBlock, StandardTransformerLM
from tensor_mem import Layer, MemoryConfig, TensorMemory, TensorMemoryLM


# =============================================================================
# Experiment Configuration
# =============================================================================

DEVICE = "cuda"
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 4
D_FF = 1024
VOCAB_SIZE = 10000
HEAD_DIM = D_MODEL // NUM_HEADS

MAX_EPOCHS = 50
PATIENCE = 2
SEQ_LEN = 64
BATCH_SIZE = 32
LR = 1e-3
CLIP = 0.5

DATA_FRACTION = 1.0


# =============================================================================
# Memory Configuration
# =============================================================================

MEMORY_CONFIG = MemoryConfig(
    dim=HEAD_DIM,
    eps=1e-6,
    use_delta_rule=False,
    max_delta=10.0,
    max_memory=100.0,
    max_norm=1000.0,
)


# =============================================================================
# Model Definitions - Declarative Configuration
# =============================================================================

# TensorMemoryLM: 4 layers, 4 heads each
TENSOR_MEMORY_MODEL = TensorMemoryLM(
    vocab_size=VOCAB_SIZE,
    layers=[
        Layer(
            [TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG)],
            hidden_size=D_MODEL,
            d_ff=D_FF,
            bias=True,
            normalize_qkv=False,
        ),
        Layer(
            [TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG)],
            hidden_size=D_MODEL,
            d_ff=D_FF,
            bias=True,
            normalize_qkv=False,
        ),
        Layer(
            [TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG)],
            hidden_size=D_MODEL,
            d_ff=D_FF,
            bias=True,
            normalize_qkv=False,
        ),
        Layer(
            [TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG)],
            hidden_size=D_MODEL,
            d_ff=D_FF,
            bias=True,
            normalize_qkv=False,
        ),
    ],
)

# StandardTransformerLM: 4 layers
STANDARD_TRANSFORMER_MODEL = StandardTransformerLM(
    vocab_size=VOCAB_SIZE,
    max_len=SEQ_LEN,
    layers=[
        StandardTransformerBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF),
        StandardTransformerBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF),
        StandardTransformerBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF),
        StandardTransformerBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF),
    ],
)
