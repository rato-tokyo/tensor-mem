"""Benchmarks for evaluating NoPE (No Positional Encoding) problems.

This module provides:
- OrderReversalBenchmark: Test for subject-object swap discrimination
"""

from __future__ import annotations

from benchmark.order_reversal import OrderReversalBenchmark, OrderReversalResult, print_comparison

__all__ = [
    "OrderReversalBenchmark",
    "OrderReversalResult",
    "print_comparison",
]
