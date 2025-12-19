"""Training utilities for language models."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from data import get_batch


@dataclass
class TrainingResult:
    """Result of training a model."""

    name: str
    num_params: int
    best_val_ppl: float
    best_epoch: int
    train_ppl_history: list[float]
    val_ppl_history: list[float]


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
