"""Benchmarks for evaluating NoPE (No Positional Encoding) problems.

This module provides:
- OrderReversalBenchmark: Test for subject-object swap discrimination
- BenchmarkConfig: Configuration for benchmarks
"""

from __future__ import annotations

from benchmark.order_reversal import (
    STANDARD_TRIPLETS,
    BenchmarkConfig,
    OrderReversalBenchmark,
    OrderReversalResult,
    default_benchmark_config,
    print_comparison,
)

__all__ = [
    "BenchmarkConfig",
    "OrderReversalBenchmark",
    "OrderReversalResult",
    "STANDARD_TRIPLETS",
    "default_benchmark_config",
    "print_comparison",
]
