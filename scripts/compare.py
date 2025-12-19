#!/usr/bin/env python3
"""Compare TensorMemoryLM vs StandardTransformerLM on WikiText.

This script trains both models on WikiText-2 and compares:
- Train/Val Perplexity (PPL)
- Context dependency: accuracy at different distances from context

Usage:
    python scripts/compare.py --device cuda
    python scripts/compare.py --device cuda --max-epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from config import default_memory_config

from analysis import ContextDependencyResult, analyze_context_dependency, print_results
from baseline import StandardTransformerBlock, StandardTransformerLM
from data import batchify, build_vocab, download_wikitext2, tokenize
from tensor_mem import Layer, TensorMemory, TensorMemoryLM
from training import TrainingResult, train_model


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare models on WikiText-2")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--clip", type=float, default=0.5, help="Gradient clipping")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--d-ff", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument(
        "--data-fraction", type=float, default=1.0, help="Fraction of data to use (0.25 = 1/4)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(
        f"Architecture: d_model={args.d_model}, heads={args.num_heads}, "
        f"layers={args.num_layers}, d_ff={args.d_ff}"
    )
    print(
        f"Training: max_epochs={args.max_epochs}, patience={args.patience}, "
        f"seq_len={args.seq_len}, batch_size={args.batch_size}"
    )

    # Load WikiText-2
    train_text, val_text, _ = download_wikitext2()

    # Build vocabulary from train data only (no data leak)
    print("\nBuilding vocabulary...")
    vocab, inv_vocab = build_vocab(train_text, args.vocab_size)
    actual_vocab_size = len(vocab)
    print(f"Vocabulary size: {actual_vocab_size}")

    # Tokenize
    print("Tokenizing...")
    train_tokens = tokenize(train_text, vocab)
    val_tokens = tokenize(val_text, vocab)

    # Apply data fraction
    if args.data_fraction < 1.0:
        train_tokens = train_tokens[: int(len(train_tokens) * args.data_fraction)]
        val_tokens = val_tokens[: int(len(val_tokens) * args.data_fraction)]
        print(f"Using {args.data_fraction:.0%} of data")

    print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")

    # Batchify
    train_data = batchify(train_tokens, args.batch_size, device)
    val_data = batchify(val_tokens, args.batch_size, device)
    print(f"Train batches: {train_data.size(0)}, Val batches: {val_data.size(0)}")

    results: list[TrainingResult] = []
    context_results: list[ContextDependencyResult] = []

    # Train TensorMemoryLM
    print("\n" + "=" * 70)
    print("Training: TensorMemoryLM (NoPE)")
    print("=" * 70)

    # Declarative Configuration: structure is visible here
    head_dim = args.d_model // args.num_heads
    memory_config = default_memory_config(dim=head_dim)

    tensor_model = TensorMemoryLM(
        vocab_size=actual_vocab_size,
        layers=[
            Layer(
                [TensorMemory(memory_config) for _ in range(args.num_heads)],
                hidden_size=args.d_model,
                d_ff=args.d_ff,
                bias=True,
                normalize_qkv=False,
            )
            for _ in range(args.num_layers)
        ],
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in tensor_model.parameters()):,}")

    tensor_result, tensor_model = train_model(
        model=tensor_model,
        name="TensorMemoryLM (NoPE)",
        train_data=train_data,
        val_data=val_data,
        max_epochs=args.max_epochs,
        seq_len=args.seq_len,
        lr=args.lr,
        clip=args.clip,
        patience=args.patience,
        has_memory=True,
    )
    results.append(tensor_result)

    tensor_context = analyze_context_dependency(
        tensor_model, val_data, args.seq_len, has_memory=True, name="TensorMemoryLM"
    )
    context_results.append(tensor_context)

    # Train StandardTransformerLM
    print("\n" + "=" * 70)
    print("Training: StandardTransformerLM (PE)")
    print("=" * 70)

    # Declarative Configuration: structure is visible here
    standard_model = StandardTransformerLM(
        vocab_size=actual_vocab_size,
        max_len=args.seq_len,
        layers=[
            StandardTransformerBlock(
                d_model=args.d_model,
                num_heads=args.num_heads,
                d_ff=args.d_ff,
            )
            for _ in range(args.num_layers)
        ],
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in standard_model.parameters()):,}")

    standard_result, standard_model = train_model(
        model=standard_model,
        name="StandardTransformerLM (PE)",
        train_data=train_data,
        val_data=val_data,
        max_epochs=args.max_epochs,
        seq_len=args.seq_len,
        lr=args.lr,
        clip=args.clip,
        patience=args.patience,
        has_memory=False,
    )
    results.append(standard_result)

    standard_context = analyze_context_dependency(
        standard_model, val_data, args.seq_len, has_memory=False, name="StandardTransformerLM"
    )
    context_results.append(standard_context)

    # Print results
    print_results(results, context_results)


if __name__ == "__main__":
    main()
