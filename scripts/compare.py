#!/usr/bin/env python3
"""Compare TensorMemoryLM vs StandardTransformerLM on WikiText-2.

This script trains both models on WikiText-2 and compares:
- Train/Val Perplexity (PPL)
- Context dependency: accuracy at different distances from context

Configuration is defined in config.py. No CLI parameters.

Usage:
    python scripts/compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from analysis import ContextDependencyResult, analyze_context_dependency, print_results
from baseline import StandardTransformerBlock, StandardTransformerLM
from config import EXPERIMENT_CONFIG, default_memory_config
from data import batchify, build_vocab, download_wikitext2, tokenize
from tensor_mem import Layer, TensorMemory, TensorMemoryLM
from training import TrainingResult, train_model


def main() -> None:
    """Main function."""
    cfg = EXPERIMENT_CONFIG

    device = torch.device(cfg.device)
    print(f"Device: {device}")
    print(
        f"Architecture: d_model={cfg.d_model}, heads={cfg.num_heads}, "
        f"layers={cfg.num_layers}, d_ff={cfg.d_ff}"
    )
    print(
        f"Training: max_epochs={cfg.max_epochs}, patience={cfg.patience}, "
        f"seq_len={cfg.seq_len}, batch_size={cfg.batch_size}"
    )

    # Load WikiText-2
    train_text, val_text, _ = download_wikitext2()

    # Build vocabulary from train data only (no data leak)
    print("\nBuilding vocabulary...")
    vocab, inv_vocab = build_vocab(train_text, cfg.vocab_size)
    actual_vocab_size = len(vocab)
    print(f"Vocabulary size: {actual_vocab_size}")

    # Tokenize
    print("Tokenizing...")
    train_tokens = tokenize(train_text, vocab)
    val_tokens = tokenize(val_text, vocab)

    # Apply data fraction
    if cfg.data_fraction < 1.0:
        train_tokens = train_tokens[: int(len(train_tokens) * cfg.data_fraction)]
        val_tokens = val_tokens[: int(len(val_tokens) * cfg.data_fraction)]
        print(f"Using {cfg.data_fraction:.0%} of data")

    print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")

    # Batchify
    train_data = batchify(train_tokens, cfg.batch_size, device)
    val_data = batchify(val_tokens, cfg.batch_size, device)
    print(f"Train batches: {train_data.size(0)}, Val batches: {val_data.size(0)}")

    results: list[TrainingResult] = []
    context_results: list[ContextDependencyResult] = []

    # =========================================================================
    # TensorMemoryLM - Declarative Configuration
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training: TensorMemoryLM (NoPE)")
    print("=" * 70)

    memory_config = default_memory_config(dim=cfg.head_dim)

    # Declarative Configuration: 4 layers, 4 heads each - structure is visible
    tensor_model = TensorMemoryLM(
        vocab_size=actual_vocab_size,
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
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in tensor_model.parameters()):,}")

    tensor_result, tensor_model = train_model(
        model=tensor_model,
        name="TensorMemoryLM (NoPE)",
        train_data=train_data,
        val_data=val_data,
        max_epochs=cfg.max_epochs,
        seq_len=cfg.seq_len,
        lr=cfg.lr,
        clip=cfg.clip,
        patience=cfg.patience,
        has_memory=True,
    )
    results.append(tensor_result)

    tensor_context = analyze_context_dependency(
        tensor_model, val_data, cfg.seq_len, has_memory=True, name="TensorMemoryLM"
    )
    context_results.append(tensor_context)

    # =========================================================================
    # StandardTransformerLM - Declarative Configuration
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training: StandardTransformerLM (PE)")
    print("=" * 70)

    # Declarative Configuration: 4 layers - structure is visible
    standard_model = StandardTransformerLM(
        vocab_size=actual_vocab_size,
        max_len=cfg.seq_len,
        layers=[
            StandardTransformerBlock(d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff),
            StandardTransformerBlock(d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff),
            StandardTransformerBlock(d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff),
            StandardTransformerBlock(d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff),
        ],
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in standard_model.parameters()):,}")

    standard_result, standard_model = train_model(
        model=standard_model,
        name="StandardTransformerLM (PE)",
        train_data=train_data,
        val_data=val_data,
        max_epochs=cfg.max_epochs,
        seq_len=cfg.seq_len,
        lr=cfg.lr,
        clip=cfg.clip,
        patience=cfg.patience,
        has_memory=False,
    )
    results.append(standard_result)

    standard_context = analyze_context_dependency(
        standard_model, val_data, cfg.seq_len, has_memory=False, name="StandardTransformerLM"
    )
    context_results.append(standard_context)

    # Print results
    print_results(results, context_results)


if __name__ == "__main__":
    main()
