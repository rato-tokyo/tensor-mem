"""TensorMemory implementation."""

from __future__ import annotations

import torch

from .base import BaseTensorMemory
from .config import MemoryConfig


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

    For multi-head attention, create multiple TensorMemory instances.

    Uses config-based initialization - no default arguments.

    Example:
        >>> from tensor_mem.memory.config import MemoryConfig
        >>> config = MemoryConfig(dim=64, eps=1e-6, use_delta_rule=False)
        >>> memory = TensorMemory(config)
        >>> memory.reset(device="cuda", dtype=torch.float16)
        >>>
        >>> keys = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        >>> values = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        >>> memory.update(keys, values)
        >>>
        >>> queries = torch.randn(4, 32, 64, device="cuda", dtype=torch.float16)
        >>> output = memory.retrieve(queries)  # [4, 32, 64]
    """

    def __init__(self, config: MemoryConfig) -> None:
        """Initialize TensorMemory.

        Args:
            config: MemoryConfig containing all memory settings.
        """
        super().__init__(config)

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
