#!/usr/bin/env python3
"""Minimal training script for TensorMemoryLM with synthetic data.

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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from tensor_mem import small_model


def create_synthetic_dataset(
    vocab_size: int,
    seq_length: int,
    num_samples: int,
    pattern_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic dataset with repeating patterns.

    Task: Learn to predict the next token in a repeating sequence.
    Example pattern [1, 2, 3, 4] -> input: [1, 2, 3, 4, 1, 2, 3], target: [2, 3, 4, 1, 2, 3, 4]

    Args:
        vocab_size: Size of vocabulary (tokens 0 to vocab_size-1).
        seq_length: Length of each sequence.
        num_samples: Number of training samples.
        pattern_length: Length of the repeating pattern.

    Returns:
        Tuple of (inputs, targets) tensors of shape [num_samples, seq_length].
    """
    inputs = []
    targets = []

    for _ in range(num_samples):
        # Create a random repeating pattern (use tokens 1 to vocab_size-1, reserve 0 for padding)
        pattern = torch.randint(1, vocab_size, (pattern_length,))

        # Repeat pattern to fill sequence
        repeats = (seq_length + pattern_length) // pattern_length + 1
        full_seq = pattern.repeat(repeats)[: seq_length + 1]

        inputs.append(full_seq[:-1])
        targets.append(full_seq[1:])

    return torch.stack(inputs), torch.stack(targets)


def train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Perform a single training step.

    Args:
        model: The model to train.
        inputs: Input tensor [batch, seq].
        targets: Target tensor [batch, seq].
        optimizer: Optimizer.
        criterion: Loss function.

    Returns:
        Loss value as float.
    """
    model.train()
    optimizer.zero_grad()

    # Reset memory for each batch (important for TensorMemoryLM)
    model.reset_memory()

    logits = model(inputs)  # [batch, seq, vocab]
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Evaluate model on dataset.

    Args:
        model: The model to evaluate.
        inputs: Input tensor [batch, seq].
        targets: Target tensor [batch, seq].
        criterion: Loss function.

    Returns:
        Tuple of (loss, accuracy).
    """
    model.eval()
    model.reset_memory()

    with torch.no_grad():
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total

    return loss.item(), accuracy


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

    # Create model using Declarative Configuration
    print("\nCreating model...")
    model = small_model(vocab_size=args.vocab_size, memory_type="standard").to(device)

    # Get structure info from the actual model
    num_layers = len(model.layers)
    num_heads = model.layers[0].attention.num_heads
    d_model = model.d_model

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
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
        # Shuffle training data
        perm = torch.randperm(len(train_inputs))
        train_inputs_shuffled = train_inputs[perm]
        train_targets_shuffled = train_targets[perm]

        # Train on batches
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_inputs), args.batch_size):
            batch_inputs = train_inputs_shuffled[i : i + args.batch_size]
            batch_targets = train_targets_shuffled[i : i + args.batch_size]

            loss = train_step(model, batch_inputs, batch_targets, optimizer, criterion)
            total_loss += loss
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Evaluate periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            eval_loss, eval_acc = evaluate(model, eval_inputs, eval_targets, criterion)
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Eval Loss: {eval_loss:.4f} | "
                f"Eval Acc: {eval_acc:.2%}"
            )

    print("-" * 60)

    # Final evaluation
    print("\nFinal evaluation:")
    eval_loss, eval_acc = evaluate(model, eval_inputs, eval_targets, criterion)
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
