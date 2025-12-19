"""Utility functions for tensor-mem."""

from __future__ import annotations

import torch
import torch.nn.functional as f


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """
    ELU + 1 activation function for linear attention.

    This activation function ensures all values are positive, which is
    essential for the normalization term in linear attention.

    φ(x) = ELU(x) + 1 = max(0, x) + min(0, exp(x) - 1) + 1

    Properties:
        - All outputs are positive (>= 1 for x >= 0, > 0 for x < 0)
        - Smooth and differentiable
        - Computationally efficient

    Args:
        x: Input tensor of any shape.

    Returns:
        Activated tensor with same shape as input, all values positive.

    Example:
        >>> x = torch.randn(2, 4, 64)
        >>> y = elu_plus_one(x)
        >>> assert (y > 0).all()
    """
    return f.elu(x) + 1.0


def repeat_kv(
    hidden_states: torch.Tensor,
    n_rep: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Repeat key-value heads for Grouped Query Attention (GQA).

    In GQA, there are fewer KV heads than query heads. This function
    repeats the KV heads to match the number of query heads.

    Args:
        hidden_states: Key or value tensor of shape [batch, seq, num_kv_heads * head_dim].
        n_rep: Number of times to repeat each KV head.
        head_dim: Dimension of each attention head.

    Returns:
        Repeated tensor of shape [batch, seq, num_kv_heads * n_rep * head_dim].

    Example:
        >>> # 4 KV heads, repeat 2x to get 8 query heads
        >>> kv = torch.randn(2, 128, 4 * 64)  # [batch, seq, 4 * 64]
        >>> expanded = repeat_kv(kv, n_rep=2, head_dim=64)
        >>> assert expanded.shape == (2, 128, 8 * 64)
    """
    if n_rep == 1:
        return hidden_states

    batch, seq_len, hidden_size = hidden_states.shape
    num_kv_heads = hidden_size // head_dim

    # Reshape to [batch, seq, num_kv_heads, head_dim]
    hidden_states = hidden_states.view(batch, seq_len, num_kv_heads, head_dim)

    # Expand to [batch, seq, num_kv_heads, n_rep, head_dim]
    hidden_states = hidden_states.unsqueeze(3).expand(batch, seq_len, num_kv_heads, n_rep, head_dim)

    # Reshape back to [batch, seq, num_kv_heads * n_rep * head_dim]
    return hidden_states.reshape(batch, seq_len, num_kv_heads * n_rep * head_dim)
