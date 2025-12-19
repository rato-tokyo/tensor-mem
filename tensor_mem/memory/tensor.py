"""TensorMemory implementation."""

from __future__ import annotations

import torch

from .base import BaseTensorMemory


class TensorMemory(BaseTensorMemory):
    """
    Tensor Product Memory (single memory unit).

    A simple associative memory that stores key-value bindings using
    outer products. This is the fundamental building block.

    Mathematical formulation:
        Update: M = M + σ(K)^T @ V / (batch * seq)
                z = z + Σσ(K) / batch
        Retrieve: output = (σ(Q) @ M) / (σ(Q) @ z + eps)

    Where σ = ELU + 1 activation function.

    Delta Rule (optional):
        When use_delta_rule=True, subtracts existing bindings before update:
        delta_v = V - retrieve(K)
        M = M + σ(K)^T @ delta_v / (batch * seq)
        This prevents duplicate bindings and improves long-context performance.

    Numerical Stability:
        - delta_m is clamped to [-max_delta, max_delta] to prevent overflow
        - M is clamped to [-max_memory, max_memory] to prevent accumulation explosion
        - z is clamped to [eps, max_norm] to ensure valid normalization

    For multi-head attention, create multiple TensorMemory instances.

    Args:
        dim: Dimension of the memory vectors.
        eps: Small constant for numerical stability.
        use_delta_rule: Whether to use Delta Rule for updates.
        max_delta: Maximum absolute value for update deltas (prevents overflow).
        max_memory: Maximum absolute value for memory matrix M.
        max_norm: Maximum value for normalization term z.

    Example:
        >>> memory = TensorMemory(dim=64)
        >>> memory.reset(device="cuda", dtype=torch.float16)
        >>>
        >>> keys = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        >>> values = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        >>> memory.update(keys, values)
        >>>
        >>> queries = torch.randn(4, 32, 64, device="cuda", dtype=torch.float16)
        >>> output = memory.retrieve(queries)  # [4, 32, 64]
    """

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Update memory with new key-value pairs (accumulative).

        Args:
            keys: Key tensor of shape [batch, seq, dim].
            values: Value tensor of shape [batch, seq, dim].
        """
        if not self.is_initialized:
            raise RuntimeError("Memory not initialized. Call reset() first.")

        _, delta_m, delta_z = self._compute_update_matrices(keys, values)

        # Accumulative update: M = M + delta_m, z = z + delta_z
        self.M = self.M + delta_m
        self.z = self.z + delta_z

        self._clamp_memory()

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"dim={self._dim}, eps={self.eps}"
