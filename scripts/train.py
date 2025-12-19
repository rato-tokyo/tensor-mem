#!/usr/bin/env python3
"""Training script for TensorMemoryLM with synthetic data.

This script demonstrates basic training on a simple sequence prediction task.
The synthetic task: predict the next token in a repeating pattern.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 100 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from config import default_memory_config
from training import count_parameters, create_synthetic_dataset, evaluate, train_epoch

from tensor_mem import Layer, TensorMemory, TensorMemoryLM


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train TensorMemoryLM on synthetic data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=64, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=32, help="Vocabulary size")
    parser.add_argument("--pattern-length", type=int, default=4, help="Repeating pattern length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--num-train", type=int, default=256, help="Number of training samples")
    parser.add_argument("--num-eval", type=int, default=64, help="Number of evaluation samples")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Training on device: {device}")

    # Declarative Configuration: structure is visible here
    print("\nCreating model...")
    config = default_memory_config(dim=64)

    model = TensorMemoryLM(
        vocab_size=args.vocab_size,
        layers=[
            Layer(
                [
                    TensorMemory(config),
                    TensorMemory(config),
                    TensorMemory(config),
                    TensorMemory(config),
                ],
                hidden_size=256,
                d_ff=1024,
                bias=True,
                normalize_qkv=False,
            ),
            Layer(
                [
                    TensorMemory(config),
                    TensorMemory(config),
                    TensorMemory(config),
                    TensorMemory(config),
                ],
                hidden_size=256,
                d_ff=1024,
                bias=True,
                normalize_qkv=False,
            ),
            Layer(
                [
                    TensorMemory(config),
                    TensorMemory(config),
                    TensorMemory(config),
                    TensorMemory(config),
                ],
                hidden_size=256,
                d_ff=1024,
                bias=True,
                normalize_qkv=False,
            ),
            Layer(
                [
                    TensorMemory(config),
                    TensorMemory(config),
                    TensorMemory(config),
                    TensorMemory(config),
                ],
                hidden_size=256,
                d_ff=1024,
                bias=True,
                normalize_qkv=False,
            ),
        ],
    ).to(device)

    num_layers = len(model.layers)
    num_heads = model.layers[0].attention.num_heads
    d_model = model.d_model

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Structure: d_model={d_model}, heads={num_heads}, layers={num_layers}")

    # Create datasets
    print("\nCreating synthetic datasets...")
    train_inputs, train_targets = create_synthetic_dataset(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        num_samples=args.num_train,
        pattern_length=args.pattern_length,
    )
    eval_inputs, eval_targets = create_synthetic_dataset(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        num_samples=args.num_eval,
        pattern_length=args.pattern_length,
    )

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    eval_inputs = eval_inputs.to(device)
    eval_targets = eval_targets.to(device)

    print(f"Train samples: {len(train_inputs)}, Eval samples: {len(eval_inputs)}")
    print(f"Sequence length: {args.seq_length}, Pattern length: {args.pattern_length}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(args.epochs):
        avg_train_loss = train_epoch(
            model,
            train_inputs,
            train_targets,
            optimizer,
            criterion,
            args.batch_size,
            has_memory=True,
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            eval_loss, eval_acc = evaluate(
                model, eval_inputs, eval_targets, criterion, has_memory=True
            )
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Eval Loss: {eval_loss:.4f} | "
                f"Eval Acc: {eval_acc:.2%}"
            )

    print("-" * 60)

    # Final evaluation
    print("\nFinal evaluation:")
    eval_loss, eval_acc = evaluate(model, eval_inputs, eval_targets, criterion, has_memory=True)
    print(f"  Loss: {eval_loss:.4f}")
    print(f"  Accuracy: {eval_acc:.2%}")

    # Show example predictions
    print("\nExample predictions (first 3 samples):")
    model.eval()
    model.reset_memory()

    with torch.no_grad():
        sample_inputs = eval_inputs[:3]
        sample_targets = eval_targets[:3]
        logits = model(sample_inputs)
        predictions = logits.argmax(dim=-1)

        for i in range(3):
            inp = sample_inputs[i, :8].tolist()
            tgt = sample_targets[i, :8].tolist()
            pred = predictions[i, :8].tolist()
            print(f"  Input:      {inp}")
            print(f"  Target:     {tgt}")
            print(f"  Prediction: {pred}")
            print()


if __name__ == "__main__":
    main()
