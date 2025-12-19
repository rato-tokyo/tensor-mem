#!/usr/bin/env python3
"""Compare TensorMemoryLM vs StandardTransformerLM on WikiText-2.

This script trains both models on WikiText-2 and compares:
- Train/Val Perplexity (PPL)
- Context dependency: accuracy at different distances from context

Configuration and model definitions: models.py

Usage:
    python scripts/compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from analysis import ContextDependencyResult, analyze_context_dependency, print_results
from data import batchify, build_vocab, download_wikitext2, tokenize
from models import (
    BATCH_SIZE,
    CLIP,
    DATA_FRACTION,
    DEVICE,
    D_FF,
    D_MODEL,
    LR,
    MAX_EPOCHS,
    NUM_HEADS,
    NUM_LAYERS,
    PATIENCE,
    SEQ_LEN,
    STANDARD_TRANSFORMER_MODEL,
    TENSOR_MEMORY_MODEL,
    VOCAB_SIZE,
)
from training import TrainingResult, train_model


def main() -> None:
    """Main function."""
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(
        f"Architecture: d_model={D_MODEL}, heads={NUM_HEADS}, "
        f"layers={NUM_LAYERS}, d_ff={D_FF}"
    )
    print(
        f"Training: max_epochs={MAX_EPOCHS}, patience={PATIENCE}, "
        f"seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}"
    )

    # Load WikiText-2
    train_text, val_text, _ = download_wikitext2()

    # Build vocabulary from train data only (no data leak)
    print("\nBuilding vocabulary...")
    vocab, inv_vocab = build_vocab(train_text, VOCAB_SIZE)
    actual_vocab_size = len(vocab)
    print(f"Vocabulary size: {actual_vocab_size}")

    # Tokenize
    print("Tokenizing...")
    train_tokens = tokenize(train_text, vocab)
    val_tokens = tokenize(val_text, vocab)

    # Apply data fraction
    if DATA_FRACTION < 1.0:
        train_tokens = train_tokens[: int(len(train_tokens) * DATA_FRACTION)]
        val_tokens = val_tokens[: int(len(val_tokens) * DATA_FRACTION)]
        print(f"Using {DATA_FRACTION:.0%} of data")

    print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")

    # Batchify
    train_data = batchify(train_tokens, BATCH_SIZE, device)
    val_data = batchify(val_tokens, BATCH_SIZE, device)
    print(f"Train batches: {train_data.size(0)}, Val batches: {val_data.size(0)}")

    results: list[TrainingResult] = []
    context_results: list[ContextDependencyResult] = []

    # =========================================================================
    # TensorMemoryLM
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training: TensorMemoryLM (NoPE)")
    print("=" * 70)

    tensor_model = TENSOR_MEMORY_MODEL.to(device)

    print(f"Parameters: {sum(p.numel() for p in tensor_model.parameters()):,}")

    tensor_result, tensor_model = train_model(
        model=tensor_model,
        name="TensorMemoryLM (NoPE)",
        train_data=train_data,
        val_data=val_data,
        max_epochs=MAX_EPOCHS,
        seq_len=SEQ_LEN,
        lr=LR,
        clip=CLIP,
        patience=PATIENCE,
        has_memory=True,
    )
    results.append(tensor_result)

    tensor_context = analyze_context_dependency(
        tensor_model, val_data, SEQ_LEN, has_memory=True, name="TensorMemoryLM"
    )
    context_results.append(tensor_context)

    # =========================================================================
    # StandardTransformerLM
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training: StandardTransformerLM (PE)")
    print("=" * 70)

    standard_model = STANDARD_TRANSFORMER_MODEL.to(device)

    print(f"Parameters: {sum(p.numel() for p in standard_model.parameters()):,}")

    standard_result, standard_model = train_model(
        model=standard_model,
        name="StandardTransformerLM (PE)",
        train_data=train_data,
        val_data=val_data,
        max_epochs=MAX_EPOCHS,
        seq_len=SEQ_LEN,
        lr=LR,
        clip=CLIP,
        patience=PATIENCE,
        has_memory=False,
    )
    results.append(standard_result)

    standard_context = analyze_context_dependency(
        standard_model, val_data, SEQ_LEN, has_memory=False, name="StandardTransformerLM"
    )
    context_results.append(standard_context)

    # Print results
    print_results(results, context_results)


if __name__ == "__main__":
    main()
