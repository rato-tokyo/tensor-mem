#!/usr/bin/env python3
"""Run order reversal benchmark comparing Standard Transformer vs Tensor Memory LLM.

Usage:
    python benchmark/run_order_reversal.py
    python benchmark/run_order_reversal.py --device cuda
"""

from __future__ import annotations

import torch

from baseline import create_standard_transformer_lm, create_tensor_memory_lm, small_config
from benchmark.order_reversal import OrderReversalBenchmark, print_comparison


def main():
    device = torch.device("cpu")
    print(f"Running on device: {device}")

    # Create benchmark
    benchmark = OrderReversalBenchmark()
    vocab_size = len(benchmark.vocab)

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Test triplets: {len(benchmark.triplets)}")

    # Create models using centralized config
    print("\nInitializing models...")

    # Get small config for TensorMemoryLM
    config = small_config(vocab_size=vocab_size, memory_type="standard")

    print(
        f"Model config: d_model={config.d_model}, heads={config.num_heads}, layers={config.num_layers}"
    )

    # Create standard transformer with explicit parameters
    standard_model = create_standard_transformer_lm(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.block.d_ff,
        max_len=512,
        dropout=config.dropout,
    )

    # Create tensor memory LM from config
    tensor_mem_model = create_tensor_memory_lm(config)

    # Count parameters
    std_params = sum(p.numel() for p in standard_model.parameters())
    mem_params = sum(p.numel() for p in tensor_mem_model.parameters())
    print(f"Standard Transformer params: {std_params:,}")
    print(f"Tensor Memory LM params: {mem_params:,}")

    # Run benchmark
    print("\nRunning benchmark...")
    models = {
        "Standard Transformer (PE)": standard_model,
        "Tensor Memory LM (NoPE)": tensor_mem_model,
    }

    results = benchmark.compare_models(models, device=device)

    # Print results
    print_comparison(results)

    # Detailed per-triplet analysis
    print("\nDetailed per-triplet similarities:")
    print("-" * 60)
    print(f"{'Triplet':<35} {'Standard':<12} {'TensorMem'}")
    print("-" * 60)

    std_result = results["Standard Transformer (PE)"]
    mem_result = results["Tensor Memory LM (NoPE)"]

    for i, (s, v, o) in enumerate(benchmark.triplets):
        triplet_str = f"{s} {v} {o}"
        print(
            f"{triplet_str:<35} "
            f"{std_result.similarities[i]:<12.4f} "
            f"{mem_result.similarities[i]:.4f}"
        )

    print("-" * 60)

    # Interpretation
    print("\nINTERPRETATION:")
    print("-" * 60)

    std_disc = std_result.discrimination_score
    mem_disc = mem_result.discrimination_score

    if std_disc > mem_disc:
        diff = std_disc - mem_disc
        print(f"Standard Transformer discriminates {diff:.2%} better")
        print("  -> Positional encoding helps distinguish word order")
    elif mem_disc > std_disc:
        diff = mem_disc - std_disc
        print(f"Tensor Memory LM discriminates {diff:.2%} better")
        print("  -> Memory mechanism provides implicit order information")
    else:
        print("Both models have similar discrimination ability")

    if mem_result.mean_similarity > 0.95:
        print("\nWARNING: Tensor Memory LM shows high similarity (>0.95)")
        print("   This indicates potential NoPE problem - model may not")
        print("   distinguish word order effectively without training.")

    print()


if __name__ == "__main__":
    main()
