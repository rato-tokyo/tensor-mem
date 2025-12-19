"""Shared training utilities for scripts.

This module provides common training functions used by train.py and compare.py.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class LMModel(Protocol):
    """Protocol for language models."""

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def train(self, mode: bool = True) -> nn.Module: ...
    def eval(self) -> nn.Module: ...
    def parameters(self) -> ...: ...


class MemoryLMModel(LMModel, Protocol):
    """Protocol for language models with memory."""

    def reset_memory(self) -> None: ...


def create_synthetic_dataset(
    vocab_size: int,
    seq_length: int,
    num_samples: int,
    pattern_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic dataset with repeating patterns.

    Task: Learn to predict the next token in a repeating sequence.
    Example pattern [1, 2, 3, 4] -> input: [1, 2, 3, 4, 1, 2, 3], target: [2, 3, 4, 1, 2, 3, 4]

    Args:
        vocab_size: Size of vocabulary (tokens 0 to vocab_size-1).
        seq_length: Length of each sequence.
        num_samples: Number of training samples.
        pattern_length: Length of the repeating pattern.

    Returns:
        Tuple of (inputs, targets) tensors of shape [num_samples, seq_length].
    """
    inputs = []
    targets = []

    for _ in range(num_samples):
        pattern = torch.randint(1, vocab_size, (pattern_length,))
        repeats = (seq_length + pattern_length) // pattern_length + 1
        full_seq = pattern.repeat(repeats)[: seq_length + 1]

        inputs.append(full_seq[:-1])
        targets.append(full_seq[1:])

    return torch.stack(inputs), torch.stack(targets)


def create_associative_recall_dataset(
    num_keys: int,
    num_samples: int,
    num_pairs: int,
    num_queries: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create associative recall dataset.

    Task: Remember key-value pairs and recall the value when given the key.

    Sequence structure:
        [K1, V1, K2, V2, ..., Kn, Vn, SEP, Q1, ?, Q2, ?, ...]

    Where:
        - Ki: Key token (from key vocabulary 1 to num_keys)
        - Vi: Value token (from value vocabulary num_keys+1 to 2*num_keys)
        - SEP: Separator token (0)
        - Qi: Query key (one of the keys seen before)
        - ?: Position where model should predict the corresponding value

    Example (num_keys=10, num_pairs=3, num_queries=2):
        Keys: 1-10, Values: 11-20, SEP: 0
        Input:  [3, 15, 7, 12, 1, 18, 0, 3, ?, 7, ?]
        Target: [*, *, *, *, *, *, *, *, 15, *, 12]
        (* = don't care, only query positions matter)

    Args:
        num_keys: Number of unique keys (vocabulary = 2*num_keys + 1).
        num_samples: Number of training samples.
        num_pairs: Number of key-value pairs per sample.
        num_queries: Number of queries per sample.

    Returns:
        Tuple of (inputs, targets, query_mask) tensors.
        - inputs: [num_samples, seq_length]
        - targets: [num_samples, seq_length]
        - query_mask: [num_samples, seq_length] boolean mask for query positions
    """
    # Vocabulary layout:
    # 0: SEP token
    # 1 to num_keys: Key tokens
    # num_keys+1 to 2*num_keys: Value tokens

    sep_token = 0
    query_placeholder = 2 * num_keys + 1  # Extra token for ? placeholder

    inputs = []
    targets = []
    query_masks = []

    for _ in range(num_samples):
        # Generate random key-value pairs (unique keys)
        keys = torch.randperm(num_keys)[:num_pairs] + 1  # Keys: 1 to num_keys
        values = torch.randint(1, num_keys + 1, (num_pairs,)) + num_keys  # Values: num_keys+1 to 2*num_keys

        # Build KV section: [K1, V1, K2, V2, ...]
        kv_section = torch.zeros(num_pairs * 2, dtype=torch.long)
        kv_section[0::2] = keys
        kv_section[1::2] = values

        # Select random queries from the keys
        query_indices = torch.randint(0, num_pairs, (num_queries,))
        query_keys = keys[query_indices]
        query_values = values[query_indices]

        # Build query section: [Q1, ?, Q2, ?, ...]
        query_section = torch.zeros(num_queries * 2, dtype=torch.long)
        query_section[0::2] = query_keys
        query_section[1::2] = query_placeholder

        # Full sequence: [KV..., SEP, Queries...]
        seq = torch.cat([kv_section, torch.tensor([sep_token]), query_section])

        # Build target (only care about positions after query keys)
        target = torch.zeros_like(seq)
        # For next-token prediction: target[i] = what should come after input[i]
        # After a query key, the next token should be the corresponding value
        query_answer_positions = len(kv_section) + 1 + torch.arange(num_queries) * 2
        target[query_answer_positions] = query_values

        # Query mask: True at positions where we evaluate
        mask = torch.zeros(len(seq), dtype=torch.bool)
        mask[query_answer_positions] = True

        inputs.append(seq)
        targets.append(target)
        query_masks.append(mask)

    return torch.stack(inputs), torch.stack(targets), torch.stack(query_masks)


def train_step(
    model: LMModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    has_memory: bool,
) -> float:
    """Perform a single training step.

    Args:
        model: The model to train.
        inputs: Input tensor [batch, seq].
        targets: Target tensor [batch, seq].
        optimizer: Optimizer.
        criterion: Loss function.
        has_memory: Whether the model has memory to reset.

    Returns:
        Loss value as float.
    """
    model.train()
    optimizer.zero_grad()

    if has_memory:
        model.reset_memory()  # type: ignore

    logits = model(inputs)
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

    loss.backward()
    optimizer.step()

    return float(loss.item())


def evaluate(
    model: LMModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    has_memory: bool,
) -> tuple[float, float]:
    """Evaluate model on dataset.

    Args:
        model: The model to evaluate.
        inputs: Input tensor [batch, seq].
        targets: Target tensor [batch, seq].
        criterion: Loss function.
        has_memory: Whether the model has memory to reset.

    Returns:
        Tuple of (loss, accuracy).
    """
    model.eval()

    if has_memory:
        model.reset_memory()  # type: ignore

    with torch.no_grad():
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total

    return loss.item(), accuracy


def train_epoch(
    model: LMModel,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    has_memory: bool,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train.
        train_inputs: Training input tensor.
        train_targets: Training target tensor.
        optimizer: Optimizer.
        criterion: Loss function.
        batch_size: Batch size.
        has_memory: Whether the model has memory.

    Returns:
        Average training loss.
    """
    perm = torch.randperm(len(train_inputs))
    train_inputs_shuffled = train_inputs[perm]
    train_targets_shuffled = train_targets[perm]

    total_loss = 0.0
    num_batches = 0

    for i in range(0, len(train_inputs), batch_size):
        batch_inputs = train_inputs_shuffled[i : i + batch_size]
        batch_targets = train_targets_shuffled[i : i + batch_size]

        loss = train_step(model, batch_inputs, batch_targets, optimizer, criterion, has_memory)
        total_loss += loss
        num_batches += 1

    return total_loss / num_batches


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters())
