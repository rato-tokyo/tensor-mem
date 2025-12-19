"""Decaying TensorMemory implementation with exponential decay."""

from __future__ import annotations

import torch

from .base import BaseTensorMemory
from .config import DecayingMemoryConfig


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

    Uses config-based initialization - no default arguments.

    Example:
        >>> from tensor_mem.memory.config import DecayingMemoryConfig
        >>> config = DecayingMemoryConfig(dim=64, eps=1e-6, use_delta_rule=False,
        ...                               max_delta=10.0, max_memory=100.0,
        ...                               max_norm=1000.0, decay=0.95)
        >>> memory = DecayingTensorMemory(config)
        >>> memory.reset(device="cuda", dtype=torch.float16)
        >>>
        >>> # Old information gradually fades with each update
        >>> for chunk in chunks:
        ...     memory.update(chunk_keys, chunk_values)
        ...     output = memory.retrieve(queries)
    """

    def __init__(self, config: DecayingMemoryConfig) -> None:
        """Initialize DecayingTensorMemory.

        Args:
            config: DecayingMemoryConfig containing all memory settings including decay.

        Raises:
            ValueError: If decay is not in range (0, 1).
        """
        if not 0.0 < config.decay < 1.0:
            raise ValueError(f"decay must be in range (0, 1), got {config.decay}")

        super().__init__(config)
        self.decay = config.decay

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

    def _extra_repr_fields(self) -> list[str]:
        """Return list of field strings for extra_repr."""
        fields = super()._extra_repr_fields()
        fields.insert(1, f"decay={self.decay}")
        return fields
