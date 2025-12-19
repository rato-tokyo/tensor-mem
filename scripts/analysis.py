"""Analysis and reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data import get_batch
from training import TrainingResult


@dataclass
class ContextDependencyResult:
    """Result of context dependency analysis."""

    name: str
    # Accuracy at different context distances
    # distance 1 = predict from 1 previous token, distance 10 = need 10+ tokens of context
    distance_accuracies: dict[str, float]


def analyze_context_dependency(
    model: nn.Module,
    data: torch.Tensor,
    seq_len: int,
    has_memory: bool,
    name: str,
) -> ContextDependencyResult:
    """Analyze how well model uses context at different distances.

    We measure accuracy at different positions within a sequence:
    - Position 0-4: Short-range dependency (1-5 tokens of context)
    - Position 5-19: Medium-range dependency (6-20 tokens of context)
    - Position 20+: Long-range dependency (20+ tokens of context)
    """
    model.eval()

    # Track correct predictions by position range
    ranges = {
        "short (1-5)": (0, 5),
        "medium (6-20)": (5, 20),
        "long (20+)": (20, seq_len),
    }

    correct_by_range: dict[str, int] = dict.fromkeys(ranges, 0)
    total_by_range: dict[str, int] = dict.fromkeys(ranges, 0)

    if has_memory:
        model.reset_memory()  # type: ignore

    with torch.no_grad():
        for i in range(0, min(data.size(0) - 1, seq_len * 100), seq_len):
            inputs, targets = get_batch(data, i, seq_len)

            if has_memory:
                model.reset_memory()  # type: ignore

            logits = model(inputs)
            predictions = logits.argmax(dim=-1)

            # Analyze by position
            for range_name, (start, end) in ranges.items():
                actual_end = min(end, inputs.size(1))
                if start >= actual_end:
                    continue

                pred_slice = predictions[:, start:actual_end]
                target_slice = targets[:, start:actual_end]

                correct = (pred_slice == target_slice).sum().item()
                total = target_slice.numel()

                correct_by_range[range_name] += correct
                total_by_range[range_name] += total

    accuracies = {
        k: correct_by_range[k] / total_by_range[k] if total_by_range[k] > 0 else 0.0 for k in ranges
    }

    return ContextDependencyResult(name=name, distance_accuracies=accuracies)


def print_results(
    results: list[TrainingResult],
    context_results: list[ContextDependencyResult],
) -> None:
    """Print comparison results."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS - WikiText-2")
    print("=" * 70)

    print(f"\n{'Model':<30} {'Params':>12} {'Best Val PPL':>14} {'Best Epoch':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r.name:<30} {r.num_params:>12,} {r.best_val_ppl:>14.2f} {r.best_epoch:>12}")

    print("-" * 70)

    # Best model
    best = min(results, key=lambda r: r.best_val_ppl)
    print(f"\nBest model: {best.name} (Val PPL={best.best_val_ppl:.2f})")

    # Training curves
    print("\n" + "-" * 70)
    print("TRAINING CURVES (Val PPL)")
    print("-" * 70)

    max_epochs = max(len(r.val_ppl_history) for r in results)
    header = f"{'Epoch':<8}"
    for r in results:
        header += f" {r.name[:18]:<18}"
    print(header)

    for i in range(min(max_epochs, 20)):  # Show first 20 epochs
        row = f"{i + 1:<8}"
        for r in results:
            if i < len(r.val_ppl_history):
                row += f" {r.val_ppl_history[i]:<18.2f}"
            else:
                row += f" {'-':<18}"
        print(row)

    if max_epochs > 20:
        print(f"  ... ({max_epochs - 20} more epochs)")

    # Context dependency
    print("\n" + "-" * 70)
    print("CONTEXT DEPENDENCY ANALYSIS")
    print("-" * 70)
    print("Accuracy at different context distances:")
    print("(Higher = better at using that amount of context)\n")

    header = f"{'Model':<25}"
    for range_name in context_results[0].distance_accuracies.keys():
        header += f" {range_name:>15}"
    print(header)
    print("-" * 70)

    for cr in context_results:
        row = f"{cr.name:<25}"
        for acc in cr.distance_accuracies.values():
            row += f" {acc:>15.2%}"
        print(row)

    print("-" * 70)
    print("\nInterpretation:")
    print("  - short (1-5): Accuracy using only 1-5 tokens of prior context")
    print("  - medium (6-20): Accuracy using 6-20 tokens of prior context")
    print("  - long (20+): Accuracy using 20+ tokens of prior context")
    print("  - Improvement from short->long shows context utilization ability")
