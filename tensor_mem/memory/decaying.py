"""Decaying TensorMemory implementation with exponential decay."""

from __future__ import annotations

import torch

from .base import BaseTensorMemory


class DecayingTensorMemory(BaseTensorMemory):
    """
    Tensor Product Memory with Exponential Decay.

    Similar to TensorMemory, but applies exponential decay to memory
    on each update. This causes older information to gradually fade,
    similar to Exponential Moving Average (EMA).

    Mathematical formulation:
        Update: M = decay * M + (1 - decay) * σ(K)^T @ V / (batch * seq)
                z = decay * z + (1 - decay) * Σσ(K) / batch
        Retrieve: output = (σ(Q) @ M) / (σ(Q) @ z + eps)

    Where σ = ELU + 1 activation function.

    The decay parameter controls how quickly old information fades:
        - decay = 0.99: Slow decay, long memory
        - decay = 0.9: Medium decay
        - decay = 0.5: Fast decay, short memory

    Args:
        dim: Dimension of the memory vectors.
        decay: Decay factor in range (0, 1). Higher = longer memory.
        eps: Small constant for numerical stability.
        use_delta_rule: Whether to use Delta Rule for updates.
        max_delta: Maximum absolute value for update deltas.
        max_memory: Maximum absolute value for memory matrix M.
        max_norm: Maximum value for normalization term z.

    Example:
        >>> memory = DecayingTensorMemory(dim=64, decay=0.95)
        >>> memory.reset(device="cuda", dtype=torch.float16)
        >>>
        >>> # Old information gradually fades with each update
        >>> for chunk in chunks:
        ...     memory.update(chunk_keys, chunk_values)
        ...     output = memory.retrieve(queries)
    """

    def __init__(
        self,
        dim: int,
        decay: float = 0.95,
        eps: float = 1e-6,
        use_delta_rule: bool = False,
        max_delta: float = 10.0,
        max_memory: float = 100.0,
        max_norm: float = 1000.0,
    ) -> None:
        """Initialize DecayingTensorMemory."""
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in range (0, 1), got {decay}")

        super().__init__(
            dim=dim,
            eps=eps,
            use_delta_rule=use_delta_rule,
            max_delta=max_delta,
            max_memory=max_memory,
            max_norm=max_norm,
        )
        self.decay = decay

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Update memory with new key-value pairs (with exponential decay).

        Args:
            keys: Key tensor of shape [batch, seq, dim].
            values: Value tensor of shape [batch, seq, dim].
        """
        if not self.is_initialized:
            raise RuntimeError("Memory not initialized. Call reset() first.")

        _, delta_m, delta_z = self._compute_update_matrices(keys, values)

        # Apply exponential decay: M = decay * M + (1 - decay) * delta_m
        self.M = self.decay * self.M + (1.0 - self.decay) * delta_m
        self.z = self.decay * self.z + (1.0 - self.decay) * delta_z

        self._clamp_memory()

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"dim={self._dim}, decay={self.decay}, eps={self.eps}"
