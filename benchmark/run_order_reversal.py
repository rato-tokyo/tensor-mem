#!/usr/bin/env python3
"""Run order reversal benchmark comparing Standard Transformer vs Tensor Memory LLM.

Usage:
    python benchmark/run_order_reversal.py
    python benchmark/run_order_reversal.py --device cuda
    python benchmark/run_order_reversal.py --d_model 512 --num_layers 6
"""

from __future__ import annotations

import argparse

import torch

from baseline.models import StandardTransformerLM, create_tensor_memory_lm
from benchmark.order_reversal import OrderReversalBenchmark, print_comparison


def main():
    parser = argparse.ArgumentParser(description="Run order reversal benchmark for NoPE analysis")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Model dimension",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=1024,
        help="Feed-forward dimension",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for distinguishability",
    )
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # Create benchmark
    benchmark = OrderReversalBenchmark(similarity_threshold=args.threshold)
    vocab_size = len(benchmark.vocab)

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Test triplets: {len(benchmark.triplets)}")
    print(f"Model config: d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}")

    # Create models
    print("\nInitializing models...")

    standard_model = StandardTransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
    )

    # Use factory function for TensorMemoryLM (DI pattern)
    tensor_mem_model = create_tensor_memory_lm(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
    )

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
