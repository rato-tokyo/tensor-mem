#!/usr/bin/env python3
"""Compare TensorMemoryLM vs StandardTransformerLM performance.

This script trains both models on the same synthetic task and compares:
- Training speed (loss convergence)
- Final accuracy (intra-sequence)
- Cross-sequence memory performance (TensorMemoryLM only)
- Parameter count

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
from training import count_parameters, create_synthetic_dataset, evaluate, train_epoch

from baseline import StandardTransformerBlock, StandardTransformerLM
from tensor_mem import Layer, TensorMemory, TensorMemoryLM


@dataclass
class TrainingResult:
    """Result of training a model."""

    name: str
    num_params: int
    final_loss: float
    final_accuracy: float
    loss_history: list[float]
    accuracy_history: list[float]


@dataclass
class ContextAnalysis:
    """Analysis of context understanding by position in pattern."""

    name: str
    position_accuracies: list[float]  # Accuracy at each position in pattern
    early_accuracy: float  # Accuracy at positions 0-1 (less context)
    late_accuracy: float  # Accuracy at positions 2+ (more context)


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


def train_model(
    model: nn.Module,
    name: str,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
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
    accuracy_history: list[float] = []

    for epoch in range(epochs):
        train_epoch(
            model,
            train_inputs,
            train_targets,
            optimizer,
            criterion,
            batch_size,
            has_memory,
        )

        if (epoch + 1) % eval_interval == 0 or epoch == 0 or epoch == epochs - 1:
            eval_loss, eval_acc = evaluate(model, eval_inputs, eval_targets, criterion, has_memory)
            loss_history.append(eval_loss)
            accuracy_history.append(eval_acc)

    final_loss, final_accuracy = evaluate(model, eval_inputs, eval_targets, criterion, has_memory)

    return TrainingResult(
        name=name,
        num_params=num_params,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        loss_history=loss_history,
        accuracy_history=accuracy_history,
    )


def analyze_context_understanding(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pattern_length: int,
    has_memory: bool,
    name: str,
) -> ContextAnalysis:
    """Analyze model's understanding of context by position in pattern.

    For a repeating pattern like [1,2,3,4,1,2,3,4,...]:
    - Position 0: Predict 2 from [1] (minimal context)
    - Position 1: Predict 3 from [1,2] (some context)
    - Position 2: Predict 4 from [1,2,3] (more context)
    - Position 3: Predict 1 from [1,2,3,4] (full pattern seen)

    Models that understand the pattern should perform better at later positions.
    """
    model.eval()
    if has_memory:
        model.reset_memory()  # type: ignore

    with torch.no_grad():
        logits = model(inputs)
        predictions = logits.argmax(dim=-1)

        # Analyze accuracy by position in pattern
        seq_len = inputs.size(1)
        position_correct = [0] * pattern_length
        position_total = [0] * pattern_length

        for pos in range(seq_len):
            pattern_pos = pos % pattern_length
            correct = (predictions[:, pos] == targets[:, pos]).sum().item()
            position_correct[pattern_pos] += correct
            position_total[pattern_pos] += inputs.size(0)

        position_accuracies = [
            position_correct[i] / position_total[i] if position_total[i] > 0 else 0.0
            for i in range(pattern_length)
        ]

        # Early (less context) vs Late (more context)
        early_correct = sum(position_correct[:2])
        early_total = sum(position_total[:2])
        late_correct = sum(position_correct[2:])
        late_total = sum(position_total[2:])

        early_accuracy = early_correct / early_total if early_total > 0 else 0.0
        late_accuracy = late_correct / late_total if late_total > 0 else 0.0

    return ContextAnalysis(
        name=name,
        position_accuracies=position_accuracies,
        early_accuracy=early_accuracy,
        late_accuracy=late_accuracy,
    )


def print_context_analysis(analyses: list[ContextAnalysis], pattern_length: int) -> None:
    """Print context understanding analysis."""
    print("\n" + "=" * 70)
    print("CONTEXT UNDERSTANDING ANALYSIS")
    print("=" * 70)
    print("\nAccuracy by position in pattern (0 = least context, 3 = most context):")
    print("-" * 70)

    # Header
    header = f"{'Model':<25}"
    for i in range(pattern_length):
        header += f" {'Pos ' + str(i):>8}"
    header += f" {'Early':>10} {'Late':>10} {'Δ':>8}"
    print(header)
    print("-" * 70)

    for a in analyses:
        row = f"{a.name:<25}"
        for acc in a.position_accuracies:
            row += f" {acc:>8.1%}"
        delta = a.late_accuracy - a.early_accuracy
        row += f" {a.early_accuracy:>10.1%} {a.late_accuracy:>10.1%} {delta:>+8.1%}"
        print(row)

    print("-" * 70)
    print("\nInterpretation:")
    print("  - Early (Pos 0-1): Predictions with less context available")
    print("  - Late (Pos 2+): Predictions with more context available")
    print("  - Δ (Delta): Improvement from having more context")
    print("  - Higher Δ = Better at using context information")


def print_comparison(
    results: list[TrainingResult], analyses: list[ContextAnalysis], pattern_length: int
) -> None:
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Model':<30} {'Params':>12} {'Loss':>10} {'Accuracy':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r.name:<30} {r.num_params:>12,} {r.final_loss:>10.4f} {r.final_accuracy:>10.2%}")

    print("-" * 70)

    # Find best model
    best_acc = max(results, key=lambda r: r.final_accuracy)
    best_loss = min(results, key=lambda r: r.final_loss)

    print(f"\nBest accuracy: {best_acc.name} ({best_acc.final_accuracy:.2%})")
    print(f"Best loss: {best_loss.name} ({best_loss.final_loss:.4f})")

    # Print training curves
    print("\n" + "-" * 70)
    print("TRAINING CURVES (Accuracy)")
    print("-" * 70)

    max_points = max(len(r.accuracy_history) for r in results)
    header = f"{'Epoch':<10}"
    for r in results:
        header += f" {r.name[:15]:<15}"
    print(header)

    for i in range(max_points):
        row = f"{i + 1:<10}"
        for r in results:
            if i < len(r.accuracy_history):
                row += f" {r.accuracy_history[i]:<15.2%}"
            else:
                row += f" {'-':<15}"
        print(row)

    # Print context analysis
    print_context_analysis(analyses, pattern_length)


def main() -> None:
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare TensorMemoryLM vs StandardTransformerLM")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=64, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=32, help="Vocabulary size")
    parser.add_argument("--pattern-length", type=int, default=4, help="Repeating pattern length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--num-train", type=int, default=256, help="Number of training samples")
    parser.add_argument("--num-eval", type=int, default=64, help="Number of evaluation samples")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(
        f"Architecture: d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}, d_ff={args.d_ff}"
    )

    # Create datasets
    print("\nCreating datasets...")
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

    print(f"Train: {len(train_inputs)}, Eval: {len(eval_inputs)}")
    print(f"Seq length: {args.seq_length}, Pattern length: {args.pattern_length}")

    results: list[TrainingResult] = []

    # Train TensorMemoryLM
    print("\n" + "=" * 70)
    print("Training: TensorMemoryLM (NoPE)")
    print("=" * 70)

    tensor_model = create_tensor_memory_model(
        vocab_size=args.vocab_size,
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
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        has_memory=True,
        eval_interval=args.eval_interval,
    )
    results.append(tensor_result)
    print(f"Final: Loss={tensor_result.final_loss:.4f}, Acc={tensor_result.final_accuracy:.2%}")

    # Train StandardTransformerLM
    print("\n" + "=" * 70)
    print("Training: StandardTransformerLM (PE)")
    print("=" * 70)

    standard_model = create_standard_model(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_len=args.seq_length,
    ).to(device)

    print(f"Parameters: {count_parameters(standard_model):,}")

    standard_result = train_model(
        model=standard_model,
        name="StandardTransformerLM (PE)",
        train_inputs=train_inputs,
        train_targets=train_targets,
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        has_memory=False,
        eval_interval=args.eval_interval,
    )
    results.append(standard_result)
    print(f"Final: Loss={standard_result.final_loss:.4f}, Acc={standard_result.final_accuracy:.2%}")

    # Context understanding analysis
    print("\n" + "=" * 70)
    print("Analyzing context understanding...")
    print("=" * 70)

    analyses: list[ContextAnalysis] = []

    tensor_analysis = analyze_context_understanding(
        model=tensor_model,
        inputs=eval_inputs,
        targets=eval_targets,
        pattern_length=args.pattern_length,
        has_memory=True,
        name="TensorMemoryLM",
    )
    analyses.append(tensor_analysis)

    standard_analysis = analyze_context_understanding(
        model=standard_model,
        inputs=eval_inputs,
        targets=eval_targets,
        pattern_length=args.pattern_length,
        has_memory=False,
        name="StandardTransformerLM",
    )
    analyses.append(standard_analysis)

    # Print comparison
    print_comparison(results, analyses, args.pattern_length)


if __name__ == "__main__":
    main()
