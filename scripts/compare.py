#!/usr/bin/env python3
"""Compare TensorMemoryLM vs StandardTransformerLM on Associative Recall task.

This script trains both models on an associative recall task and compares:
- Recall accuracy (can the model remember and retrieve key-value pairs?)
- Training speed (loss convergence)
- Parameter count

The Associative Recall task:
    Input:  [K1, V1, K2, V2, ..., Kn, Vn, SEP, Q1, ?, Q2, ?, ...]
    Output: Model should predict Vi when given Ki as query

This task directly tests the memory retrieval capability that TensorMemoryLM
is designed for.

Usage:
    python scripts/compare.py
    python scripts/compare.py --epochs 100 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from config import default_memory_config
from training import count_parameters, create_associative_recall_dataset

from baseline import StandardTransformerBlock, StandardTransformerLM
from tensor_mem import Layer, TensorMemory, TensorMemoryLM


@dataclass
class TrainingResult:
    """Result of training a model."""

    name: str
    num_params: int
    final_loss: float
    final_recall_accuracy: float
    loss_history: list[float]
    recall_accuracy_history: list[float]


def create_tensor_memory_model(
    vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int
) -> TensorMemoryLM:
    """Create TensorMemoryLM with specified architecture."""
    head_dim = d_model // num_heads
    config = default_memory_config(dim=head_dim)

    layers = [
        Layer(
            [TensorMemory(config) for _ in range(num_heads)],
            hidden_size=d_model,
            d_ff=d_ff,
            bias=True,
            normalize_qkv=False,
        )
        for _ in range(num_layers)
    ]

    return TensorMemoryLM(vocab_size=vocab_size, layers=layers)


def create_standard_model(
    vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_len: int
) -> StandardTransformerLM:
    """Create StandardTransformerLM with specified architecture."""
    layers = [
        StandardTransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        for _ in range(num_layers)
    ]

    return StandardTransformerLM(vocab_size=vocab_size, max_len=max_len, layers=layers)


def train_step_masked(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    query_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    has_memory: bool,
) -> float:
    """Training step with masked loss (only compute loss at query positions)."""
    model.train()
    optimizer.zero_grad()

    if has_memory:
        model.reset_memory()  # type: ignore

    logits = model(inputs)

    # Flatten for loss computation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    mask_flat = query_mask.view(-1)

    # Only compute loss at query positions
    masked_logits = logits_flat[mask_flat]
    masked_targets = targets_flat[mask_flat]

    loss = criterion(masked_logits, masked_targets)
    loss.backward()
    optimizer.step()

    return float(loss.item())


def evaluate_recall(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    query_mask: torch.Tensor,
    criterion: nn.Module,
    has_memory: bool,
) -> tuple[float, float]:
    """Evaluate recall accuracy at query positions only."""
    model.eval()

    if has_memory:
        model.reset_memory()  # type: ignore

    with torch.no_grad():
        logits = model(inputs)
        predictions = logits.argmax(dim=-1)

        # Flatten
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        mask_flat = query_mask.view(-1)

        # Loss at query positions
        logits_flat = logits.view(-1, logits.size(-1))
        masked_logits = logits_flat[mask_flat]
        masked_targets = targets_flat[mask_flat]
        loss = criterion(masked_logits, masked_targets).item()

        # Accuracy at query positions
        masked_predictions = predictions_flat[mask_flat]
        correct = (masked_predictions == masked_targets).sum().item()
        total = mask_flat.sum().item()
        accuracy = correct / total if total > 0 else 0.0

    return loss, accuracy


def train_model(
    model: nn.Module,
    name: str,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    train_mask: torch.Tensor,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    eval_mask: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    has_memory: bool,
    eval_interval: int,
) -> TrainingResult:
    """Train a model and return results."""
    num_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_history: list[float] = []
    recall_accuracy_history: list[float] = []

    for epoch in range(epochs):
        # Shuffle training data
        perm = torch.randperm(len(train_inputs))
        train_inputs_shuffled = train_inputs[perm]
        train_targets_shuffled = train_targets[perm]
        train_mask_shuffled = train_mask[perm]

        # Train epoch
        for i in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs_shuffled[i : i + batch_size]
            batch_targets = train_targets_shuffled[i : i + batch_size]
            batch_mask = train_mask_shuffled[i : i + batch_size]

            train_step_masked(
                model, batch_inputs, batch_targets, batch_mask, optimizer, criterion, has_memory
            )

        # Evaluate
        if (epoch + 1) % eval_interval == 0 or epoch == 0 or epoch == epochs - 1:
            eval_loss, eval_acc = evaluate_recall(
                model, eval_inputs, eval_targets, eval_mask, criterion, has_memory
            )
            loss_history.append(eval_loss)
            recall_accuracy_history.append(eval_acc)
            print(f"  Epoch {epoch + 1}: Loss={eval_loss:.4f}, Recall={eval_acc:.2%}")

    final_loss, final_accuracy = evaluate_recall(
        model, eval_inputs, eval_targets, eval_mask, criterion, has_memory
    )

    return TrainingResult(
        name=name,
        num_params=num_params,
        final_loss=final_loss,
        final_recall_accuracy=final_accuracy,
        loss_history=loss_history,
        recall_accuracy_history=recall_accuracy_history,
    )


def print_example_data(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    query_mask: torch.Tensor,
    num_keys: int,
    num_examples: int,
) -> None:
    """Print example data samples for understanding."""
    print("\n" + "=" * 70)
    print("EXAMPLE DATA SAMPLES")
    print("=" * 70)
    print(f"Vocabulary: Keys=1-{num_keys}, Values={num_keys + 1}-{2 * num_keys}, SEP=0, ?={2 * num_keys + 1}")
    print("-" * 70)

    for i in range(min(num_examples, len(inputs))):
        inp = inputs[i].tolist()
        tgt = targets[i].tolist()
        mask = query_mask[i].tolist()

        print(f"\nSample {i + 1}:")
        print(f"  Input:  {inp}")

        # Find query positions and show expected answers
        query_positions = [j for j, m in enumerate(mask) if m]
        print(f"  Query positions: {query_positions}")
        print(f"  Expected answers: {[tgt[j] for j in query_positions]}")


def print_comparison(results: list[TrainingResult]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS - ASSOCIATIVE RECALL TASK")
    print("=" * 70)
    print("\nTask: Remember [Key, Value] pairs and recall Value when given Key")
    print("-" * 70)

    print(f"\n{'Model':<30} {'Params':>12} {'Loss':>10} {'Recall Acc':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r.name:<30} {r.num_params:>12,} {r.final_loss:>10.4f} {r.final_recall_accuracy:>12.2%}")

    print("-" * 70)

    # Find best model
    best_acc = max(results, key=lambda r: r.final_recall_accuracy)
    best_loss = min(results, key=lambda r: r.final_loss)

    print(f"\nBest recall accuracy: {best_acc.name} ({best_acc.final_recall_accuracy:.2%})")
    print(f"Best loss: {best_loss.name} ({best_loss.final_loss:.4f})")

    # Print training curves
    print("\n" + "-" * 70)
    print("TRAINING CURVES (Recall Accuracy)")
    print("-" * 70)

    max_points = max(len(r.recall_accuracy_history) for r in results)
    header = f"{'Epoch':<10}"
    for r in results:
        header += f" {r.name[:15]:<15}"
    print(header)

    for i in range(max_points):
        row = f"{i + 1:<10}"
        for r in results:
            if i < len(r.recall_accuracy_history):
                row += f" {r.recall_accuracy_history[i]:<15.2%}"
            else:
                row += f" {'-':<15}"
        print(row)

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print("This task tests associative memory:")
    print("  - Model sees [K1, V1, K2, V2, ...] then queries [Q1, ?, Q2, ?]")
    print("  - Must recall the correct Value for each Query Key")
    print("  - Random baseline: 1/num_keys accuracy")
    print("  - TensorMemoryLM should excel at this memory retrieval task")


def main() -> None:
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare TensorMemoryLM vs StandardTransformerLM on Associative Recall"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-keys", type=int, default=16, help="Number of unique keys")
    parser.add_argument("--num-pairs", type=int, default=8, help="Number of KV pairs per sample")
    parser.add_argument("--num-queries", type=int, default=4, help="Number of queries per sample")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--num-train", type=int, default=512, help="Number of training samples")
    parser.add_argument("--num-eval", type=int, default=128, help="Number of evaluation samples")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluation interval")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(
        f"Architecture: d_model={args.d_model}, heads={args.num_heads}, "
        f"layers={args.num_layers}, d_ff={args.d_ff}"
    )

    # Vocabulary size: 0 (SEP) + num_keys (keys) + num_keys (values) + 1 (?)
    vocab_size = 2 * args.num_keys + 2
    seq_length = args.num_pairs * 2 + 1 + args.num_queries * 2  # KV pairs + SEP + queries

    print(f"\nTask: Associative Recall")
    print(f"  Keys: 1-{args.num_keys}, Values: {args.num_keys + 1}-{2 * args.num_keys}")
    print(f"  Pairs per sample: {args.num_pairs}, Queries per sample: {args.num_queries}")
    print(f"  Sequence length: {seq_length}, Vocabulary size: {vocab_size}")
    print(f"  Random baseline: {1 / args.num_keys:.2%}")

    # Create datasets
    print("\nCreating datasets...")
    train_inputs, train_targets, train_mask = create_associative_recall_dataset(
        num_keys=args.num_keys,
        num_samples=args.num_train,
        num_pairs=args.num_pairs,
        num_queries=args.num_queries,
    )
    eval_inputs, eval_targets, eval_mask = create_associative_recall_dataset(
        num_keys=args.num_keys,
        num_samples=args.num_eval,
        num_pairs=args.num_pairs,
        num_queries=args.num_queries,
    )

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    train_mask = train_mask.to(device)
    eval_inputs = eval_inputs.to(device)
    eval_targets = eval_targets.to(device)
    eval_mask = eval_mask.to(device)

    print(f"Train: {len(train_inputs)}, Eval: {len(eval_inputs)}")

    # Print example data
    print_example_data(train_inputs, train_targets, train_mask, args.num_keys, num_examples=2)

    results: list[TrainingResult] = []

    # Train TensorMemoryLM
    print("\n" + "=" * 70)
    print("Training: TensorMemoryLM (NoPE)")
    print("=" * 70)

    tensor_model = create_tensor_memory_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
    ).to(device)

    print(f"Parameters: {count_parameters(tensor_model):,}")

    tensor_result = train_model(
        model=tensor_model,
        name="TensorMemoryLM (NoPE)",
        train_inputs=train_inputs,
        train_targets=train_targets,
        train_mask=train_mask,
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        eval_mask=eval_mask,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        has_memory=True,
        eval_interval=args.eval_interval,
    )
    results.append(tensor_result)

    # Train StandardTransformerLM
    print("\n" + "=" * 70)
    print("Training: StandardTransformerLM (PE)")
    print("=" * 70)

    standard_model = create_standard_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_len=seq_length,
    ).to(device)

    print(f"Parameters: {count_parameters(standard_model):,}")

    standard_result = train_model(
        model=standard_model,
        name="StandardTransformerLM (PE)",
        train_inputs=train_inputs,
        train_targets=train_targets,
        train_mask=train_mask,
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        eval_mask=eval_mask,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        has_memory=False,
        eval_interval=args.eval_interval,
    )
    results.append(standard_result)

    # Print comparison
    print_comparison(results)


if __name__ == "__main__":
    main()
