"""Decaying TensorMemory implementation with exponential decay."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import elu_plus_one


class DecayingTensorMemory(nn.Module):
    """
    Tensor Product Memory with Exponential Decay.

    Similar to TensorMemory, but applies exponential decay to memory
    on each update. This causes older information to gradually fade,
    similar to Exponential Moving Average (EMA).

    Memory structure:
        M: [dim, dim] - Associative matrix storing KV bindings
        z: [dim] - Normalization term (cumulative sum of keys)

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
        super().__init__()

        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in range (0, 1), got {decay}")

        self._dim = dim
        self.decay = decay
        self.eps = eps
        self.use_delta_rule = use_delta_rule
        self.max_delta = max_delta
        self.max_memory = max_memory
        self.max_norm = max_norm

        self.register_buffer("M", None, persistent=False)
        self.register_buffer("z", None, persistent=False)

    @property
    def dim(self) -> int:
        """Return the memory dimension."""
        return self._dim

    @property
    def is_initialized(self) -> bool:
        """Check if memory buffers have been initialized."""
        return self.M is not None and self.z is not None

    @property
    def is_empty(self) -> bool:
        """Check if memory is empty (all zeros or not initialized)."""
        if not self.is_initialized:
            return True
        return self.z.sum().item() == 0

    def reset(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Reset memory to zeros.

        Args:
            device: Device to place tensors on.
            dtype: Data type for tensors.
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        self.M = torch.zeros(self._dim, self._dim, device=device, dtype=dtype)
        self.z = torch.zeros(self._dim, device=device, dtype=dtype)

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

        batch, seq, _ = keys.shape

        sigma_k = elu_plus_one(keys)

        # Delta Rule: subtract existing bindings before update
        if self.use_delta_rule and not self.is_empty:
            retrieved = torch.matmul(sigma_k, self.M)
            norm = torch.matmul(sigma_k, self.z)
            existing = retrieved / (norm.unsqueeze(-1) + self.eps)
            update_values = values - existing
        else:
            update_values = values

        # Compute new contribution: σ(K)^T @ V / (batch * seq)
        new_m = torch.einsum("bsd,bse->de", sigma_k, update_values)
        new_m = new_m / (batch * seq)

        # Clamp new contribution
        new_m = torch.clamp(new_m, min=-self.max_delta, max=self.max_delta)

        # Apply exponential decay: M = decay * M + (1 - decay) * new_m
        self.M = self.decay * self.M + (1.0 - self.decay) * new_m
        self.M = torch.clamp(self.M, min=-self.max_memory, max=self.max_memory)

        # z = decay * z + (1 - decay) * Σσ(K) / batch
        new_z = sigma_k.sum(dim=(0, 1)) / batch
        self.z = self.decay * self.z + (1.0 - self.decay) * new_z
        self.z = torch.clamp(self.z, min=self.eps, max=self.max_norm)

    def retrieve(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from memory based on queries.

        Args:
            queries: Query tensor of shape [batch, seq, dim].

        Returns:
            Retrieved values of shape [batch, seq, dim].
        """
        if not self.is_initialized:
            raise RuntimeError("Memory not initialized. Call reset() first.")

        sigma_q = elu_plus_one(queries)

        # (σ(Q) @ M)
        retrieved = torch.matmul(sigma_q, self.M)

        # σ(Q) @ z
        norm = torch.matmul(sigma_q, self.z)

        # Normalize
        output = retrieved / (norm.unsqueeze(-1) + self.eps)

        return output

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"dim={self._dim}, decay={self.decay}, eps={self.eps}"
