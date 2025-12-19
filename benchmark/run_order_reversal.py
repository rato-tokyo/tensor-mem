#!/usr/bin/env python3
"""Run order reversal benchmark comparing Standard Transformer vs Tensor Memory LLM.

Usage:
    python benchmark/run_order_reversal.py
    python benchmark/run_order_reversal.py --device cuda
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from baseline import StandardTransformerConfig, create_standard_transformer_lm
from benchmark.order_reversal import OrderReversalBenchmark, print_comparison
from tensor_mem import small_model


def main() -> None:
    device = torch.device("cpu")
    print(f"Running on device: {device}")

    # Create benchmark
    benchmark = OrderReversalBenchmark()
    vocab_size = len(benchmark.vocab)

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Test triplets: {len(benchmark.triplets)}")

    # Create models
    print("\nInitializing models...")

    # TensorMemoryLM: 4 layers, 4 heads, hidden=256, d_ff=1024
    tensor_mem_model = small_model(vocab_size=vocab_size, memory_type="standard")

    # Get structure info from the model
    num_layers = len(tensor_mem_model.layers)
    num_heads = tensor_mem_model.layers[0].attention.num_heads
    d_model = tensor_mem_model.d_model
    d_ff = tensor_mem_model.layers[0].ffn[0].out_features

    print(f"Model config: d_model={d_model}, heads={num_heads}, layers={num_layers}")

    # Create standard transformer with matching parameters
    std_config = StandardTransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=512,
        dropout=0.1,
    )
    standard_model = create_standard_transformer_lm(std_config)

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
