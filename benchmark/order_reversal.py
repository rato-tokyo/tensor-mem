"""Benchmarks for testing NoPE (No Positional Encoding) problems.

OrderReversalBenchmark: Tests if the model can distinguish subject-object swaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Special token constants
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for OrderReversalBenchmark.

    All fields required - no defaults.
    """

    similarity_threshold: float
    triplets: tuple[tuple[str, str, str], ...]


# Standard triplets for order reversal testing
STANDARD_TRIPLETS: tuple[tuple[str, str, str], ...] = (
    ("cat", "chases", "dog"),
    ("alice", "loves", "bob"),
    ("teacher", "teaches", "student"),
    ("predator", "hunts", "prey"),
    ("parent", "raises", "child"),
    ("sun", "illuminates", "moon"),
    ("fire", "melts", "ice"),
    ("hammer", "hits", "nail"),
    ("key", "opens", "door"),
    ("water", "fills", "cup"),
)


def default_benchmark_config() -> BenchmarkConfig:
    """Get default benchmark configuration."""
    return BenchmarkConfig(
        similarity_threshold=0.9,
        triplets=STANDARD_TRIPLETS,
    )


@dataclass
class OrderReversalResult:
    """Result of order reversal benchmark."""

    similarities: list[float]
    mean_similarity: float
    std_similarity: float
    discrimination_score: float
    num_distinguishable: int
    total_pairs: int

    def __repr__(self) -> str:
        return (
            f"OrderReversalResult(\n"
            f"  mean_similarity={self.mean_similarity:.4f},\n"
            f"  discrimination_score={self.discrimination_score:.4f},\n"
            f"  distinguishable={self.num_distinguishable}/{self.total_pairs}\n"
            f")"
        )


class OrderReversalBenchmark:
    """Benchmark for testing subject-object swap discrimination.

    Tests if the model produces different outputs for:
    - "X hit Y" vs "Y hit X"
    - "A loves B" vs "B loves A"
    etc.

    A model with good positional understanding should produce
    very different outputs for these reversed sentences.
    A NoPE model may produce identical outputs (similarity = 1.0).
    """

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """Initialize the benchmark.

        Args:
            config: Benchmark configuration. If None, uses default config.
        """
        if config is None:
            config = default_benchmark_config()

        self.config = config
        self.triplets = list(config.triplets)
        self.similarity_threshold = config.similarity_threshold
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> dict[str, int]:
        """Build vocabulary from triplets."""
        tokens = set()
        for s, v, o in self.triplets:
            tokens.add(s)
            tokens.add(v)
            tokens.add(o)

        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for i, token in enumerate(sorted(tokens)):
            vocab[token] = i + 2

        return vocab

    def _tokenize(self, words: list[str]) -> torch.Tensor:
        """Convert words to token IDs."""
        ids = [self.vocab.get(w, self.vocab[UNK_TOKEN]) for w in words]
        return torch.tensor(ids, dtype=torch.long)

    def _cosine_similarity(
        self,
        hidden1: torch.Tensor,
        hidden2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between sequence representations."""
        vec1 = hidden1.mean(dim=1)
        vec2 = hidden2.mean(dim=1)
        sim = F.cosine_similarity(vec1, vec2, dim=-1)
        return sim.mean().item()

    @torch.no_grad()
    def run(self, model: nn.Module, device: torch.device) -> OrderReversalResult:
        """Run the order reversal benchmark.

        Args:
            model: Language model with get_hidden_states method
            device: Device to run on

        Returns:
            OrderReversalResult with similarity metrics
        """
        model.eval()

        similarities = []
        num_distinguishable = 0

        for subject, verb, obj in self.triplets:
            original = self._tokenize([subject, verb, obj]).unsqueeze(0).to(device)
            reversed_order = self._tokenize([obj, verb, subject]).unsqueeze(0).to(device)

            if hasattr(model, "reset_memory"):
                model.reset_memory()

            hidden_orig = model.get_hidden_states(original)

            if hasattr(model, "reset_memory"):
                model.reset_memory()

            hidden_rev = model.get_hidden_states(reversed_order)

            sim = self._cosine_similarity(hidden_orig, hidden_rev)
            similarities.append(sim)

            if sim < self.similarity_threshold:
                num_distinguishable += 1

        mean_sim = sum(similarities) / len(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
        std_sim = variance**0.5

        return OrderReversalResult(
            similarities=similarities,
            mean_similarity=mean_sim,
            std_similarity=std_sim,
            discrimination_score=1.0 - mean_sim,
            num_distinguishable=num_distinguishable,
            total_pairs=len(self.triplets),
        )

    def compare_models(
        self,
        models: dict[str, nn.Module],
        device: torch.device,
    ) -> dict[str, OrderReversalResult]:
        """Run benchmark on multiple models for comparison.

        Args:
            models: Dict of model_name -> model
            device: Device to run on

        Returns:
            Dict of model_name -> OrderReversalResult
        """
        results = {}
        for name, model in models.items():
            model = model.to(device)
            results[name] = self.run(model, device)
        return results


def print_comparison(results: dict[str, OrderReversalResult]) -> None:
    """Pretty print comparison of benchmark results."""
    print("\n" + "=" * 60)
    print("ORDER REVERSAL BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Model':<25} {'Mean Sim':<12} {'Discrim':<12} {'Pass/Total'}")
    print("-" * 60)

    for name, result in sorted(results.items(), key=lambda x: -x[1].discrimination_score):
        print(
            f"{name:<25} "
            f"{result.mean_similarity:<12.4f} "
            f"{result.discrimination_score:<12.4f} "
            f"{result.num_distinguishable}/{result.total_pairs}"
        )

    print("=" * 60)
    print("Note: Lower similarity = better discrimination of word order")
    print("      Discrimination score = 1 - mean_similarity (higher is better)")
    print()
