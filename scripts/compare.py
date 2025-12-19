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
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from config import default_memory_config

from baseline import StandardTransformerBlock, StandardTransformerLM
from tensor_mem import Layer, TensorMemory, TensorMemoryLM


@dataclass
class TrainingResult:
    """Result of training a model."""

    name: str
    num_params: int
    best_val_ppl: float
    best_epoch: int
    train_ppl_history: list[float]
    val_ppl_history: list[float]


@dataclass
class ContextDependencyResult:
    """Result of context dependency analysis."""

    name: str
    # Accuracy at different context distances
    # distance 1 = predict from 1 previous token, distance 10 = need 10+ tokens of context
    distance_accuracies: dict[str, float]


def download_wikitext2() -> tuple[str, str, str]:
    """Download WikiText-2 dataset and return train/val/test text."""
    import ssl
    import urllib.request

    # Try datasets library first (most reliable for Hugging Face)
    try:
        print("Downloading WikiText-2 using datasets library...")
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        train_text = "\n".join(dataset["train"]["text"])
        val_text = "\n".join(dataset["validation"]["text"])
        test_text = "\n".join(dataset["test"]["text"])

        return train_text, val_text, test_text

    except ImportError:
        print("datasets library not available, trying alternative source...")
    except Exception as e:
        print(f"datasets library failed: {e}, trying alternative source...")

    # Fallback: Try raw GitHub (PyTorch examples)
    try:
        print("Downloading WikiText-2 from PyTorch examples...")
        base_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"

        # Create SSL context that doesn't verify (for Colab compatibility)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        def fetch(url: str) -> str:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx) as response:
                return response.read().decode("utf-8")

        train_text = fetch(f"{base_url}/train.txt")
        val_text = fetch(f"{base_url}/valid.txt")
        test_text = fetch(f"{base_url}/test.txt")

        return train_text, val_text, test_text

    except Exception as e:
        print(f"Failed to download: {e}")
        raise RuntimeError(
            "Could not download WikiText-2. Please install datasets:\n  pip install datasets"
        ) from e


def build_vocab(text: str, max_vocab: int) -> tuple[dict[str, int], dict[int, str]]:
    """Build vocabulary from text."""
    from collections import Counter

    words = text.split()
    word_counts = Counter(words)

    # Reserve 0 for <unk>, 1 for <eos>
    vocab = {"<unk>": 0, "<eos>": 1}

    for word, _ in word_counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


def tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    """Tokenize text using vocabulary."""
    unk_id = vocab["<unk>"]
    eos_id = vocab["<eos>"]

    tokens = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            for word in line.split():
                tokens.append(vocab.get(word, unk_id))
            tokens.append(eos_id)

    return tokens


def batchify(data: list[int], batch_size: int, device: torch.device) -> torch.Tensor:
    """Reshape data into [seq_len, batch_size] for language modeling."""
    nbatch = len(data) // batch_size
    data = data[: nbatch * batch_size]
    data = torch.tensor(data, dtype=torch.long, device=device)
    return data.view(batch_size, -1).t().contiguous()


def get_batch(source: torch.Tensor, i: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data for language modeling."""
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i : i + seq_len].t()  # [batch, seq_len]
    target = source[i + 1 : i + 1 + seq_len].t()  # [batch, seq_len]
    return data, target


def evaluate_ppl(
    model: nn.Module,
    data: torch.Tensor,
    seq_len: int,
    has_memory: bool,
) -> float:
    """Evaluate perplexity on data."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    if has_memory:
        model.reset_memory()  # type: ignore

    with torch.no_grad():
        for i in range(0, data.size(0) - 1, seq_len):
            inputs, targets = get_batch(data, i, seq_len)

            if has_memory:
                model.reset_memory()  # type: ignore

            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def train_epoch(
    model: nn.Module,
    train_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    seq_len: int,
    clip: float,
    has_memory: bool,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, train_data.size(0) - 1, seq_len):
        inputs, targets = get_batch(train_data, i, seq_len)

        if has_memory:
            model.reset_memory()  # type: ignore

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def train_model(
    model: nn.Module,
    name: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    max_epochs: int,
    seq_len: int,
    lr: float,
    clip: float,
    patience: int,
    has_memory: bool,
) -> tuple[TrainingResult, nn.Module]:
    """Train a model with early stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters())
    train_ppl_history: list[float] = []
    val_ppl_history: list[float] = []

    best_val_ppl = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        train_ppl = train_epoch(model, train_data, optimizer, criterion, seq_len, clip, has_memory)
        val_ppl = evaluate_ppl(model, val_data, seq_len, has_memory)

        train_ppl_history.append(train_ppl)
        val_ppl_history.append(val_ppl)

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1
            marker = ""

        print(f"  Epoch {epoch + 1}: Train PPL={train_ppl:.2f}, Val PPL={val_ppl:.2f}{marker}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    result = TrainingResult(
        name=name,
        num_params=num_params,
        best_val_ppl=best_val_ppl,
        best_epoch=best_epoch,
        train_ppl_history=train_ppl_history,
        val_ppl_history=val_ppl_history,
    )

    return result, model


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
    print("  - Improvement from short→long shows context utilization ability")


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

    # Build vocabulary
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
