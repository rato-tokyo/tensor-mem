"""Benchmarks for testing NoPE (No Positional Encoding) problems.

OrderReversalBenchmark: Tests if the model can distinguish subject-object swaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as f


class LMProtocol(Protocol):
    """Protocol for language models that can return hidden states."""

    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states for input tokens."""
        ...


@dataclass
class OrderReversalResult:
    """Result of order reversal benchmark."""

    # Per-pair cosine similarities
    similarities: list[float]

    # Average similarity (lower = better discrimination)
    mean_similarity: float

    # Standard deviation
    std_similarity: float

    # Discrimination score (1 - mean_similarity, higher = better)
    discrimination_score: float

    # Number of pairs that were distinguishable (similarity < threshold)
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
    A NoPE model may produce identical outputs (similarity ≈ 1.0).
    """

    # Default test pairs: (subject, verb, object)
    # Each pair will be tested as [S, V, O] vs [O, V, S]
    DEFAULT_TRIPLETS: list[tuple[str, str, str]] = [
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
    ]

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        triplets: list[tuple[str, str, str]] | None = None,
        similarity_threshold: float = 0.9,
    ):
        """Initialize the benchmark.

        Args:
            vocab: Token to ID mapping. If None, creates from triplets.
            triplets: List of (subject, verb, object) triplets to test.
            similarity_threshold: Threshold for considering pairs distinguishable.
        """
        self.triplets = triplets or self.DEFAULT_TRIPLETS
        self.similarity_threshold = similarity_threshold

        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab

    def _build_vocab(self) -> dict[str, int]:
        """Build vocabulary from triplets."""
        tokens = set()
        for s, v, o in self.triplets:
            tokens.add(s)
            tokens.add(v)
            tokens.add(o)

        # Add special tokens
        vocab = {"<pad>": 0, "<unk>": 1}
        for i, token in enumerate(sorted(tokens)):
            vocab[token] = i + 2

        return vocab

    def _tokenize(self, words: list[str]) -> torch.Tensor:
        """Convert words to token IDs."""
        ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in words]
        return torch.tensor(ids, dtype=torch.long)

    def _cosine_similarity(
        self,
        hidden1: torch.Tensor,
        hidden2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between sequence representations.

        Uses mean pooling over sequence dimension.

        Args:
            hidden1: Hidden states of shape (batch, seq_len, d_model)
            hidden2: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Cosine similarity as float
        """
        # Mean pool over sequence
        vec1 = hidden1.mean(dim=1)  # (batch, d_model)
        vec2 = hidden2.mean(dim=1)  # (batch, d_model)

        # Cosine similarity
        sim = f.cosine_similarity(vec1, vec2, dim=-1)
        return sim.mean().item()

    @torch.no_grad()
    def run(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
    ) -> OrderReversalResult:
        """Run the order reversal benchmark.

        Args:
            model: Language model with get_hidden_states method
            device: Device to run on

        Returns:
            OrderReversalResult with similarity metrics
        """
        model.eval()
        device = torch.device(device)

        similarities = []
        num_distinguishable = 0

        for subject, verb, obj in self.triplets:
            # Original order: [subject, verb, object]
            original = self._tokenize([subject, verb, obj]).unsqueeze(0).to(device)

            # Reversed order: [object, verb, subject]
            reversed_order = self._tokenize([obj, verb, subject]).unsqueeze(0).to(device)

            # Reset memory if model supports it
            if hasattr(model, "reset_memory"):
                model.reset_memory()

            # Get hidden states for original
            hidden_orig = model.get_hidden_states(original)

            # Reset memory again for fair comparison
            if hasattr(model, "reset_memory"):
                model.reset_memory()

            # Get hidden states for reversed
            hidden_rev = model.get_hidden_states(reversed_order)

            # Compute similarity
            sim = self._cosine_similarity(hidden_orig, hidden_rev)
            similarities.append(sim)

            if sim < self.similarity_threshold:
                num_distinguishable += 1

        # Compute statistics
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
        device: torch.device | str = "cpu",
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
    """Pretty print comparison of benchmark results.

    Args:
        results: Dict of model_name -> OrderReversalResult
    """
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
