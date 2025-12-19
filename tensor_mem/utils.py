"""Utility functions for tensor-mem."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """
    ELU + 1 activation function for linear attention.

    Ensures all values are positive, which is essential for
    the normalization term in linear attention.

    σ(x) = ELU(x) + 1 = { x + 1           if x >= 0
                       { exp(x)          if x < 0

    Args:
        x: Input tensor of any shape.

    Returns:
        Activated tensor with same shape, all values positive.
    """
    return F.elu(x) + 1.0
