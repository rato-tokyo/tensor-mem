"""Base TensorMemory implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import elu_plus_one


class TensorMemory(nn.Module):
    """
    Tensor Product Memory (single memory unit).

    A simple associative memory that stores key-value bindings using
    outer products. This is the fundamental building block.

    Memory structure:
        M: [dim, dim] - Associative matrix storing KV bindings
        z: [dim] - Normalization term (cumulative sum of keys)

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

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        use_delta_rule: bool = False,
        max_delta: float = 10.0,
        max_memory: float = 100.0,
        max_norm: float = 1000.0,
    ) -> None:
        """Initialize TensorMemory."""
        super().__init__()
        self._dim = dim
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
        Update memory with new key-value pairs.

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

        # M = M + σ(K)^T @ V / (batch * seq)
        delta_m = torch.einsum("bsd,bse->de", sigma_k, update_values)
        delta_m = delta_m / (batch * seq)

        # Clamp delta to prevent overflow (especially in fp16)
        delta_m = torch.clamp(delta_m, min=-self.max_delta, max=self.max_delta)

        # Update and clamp M to prevent accumulation explosion
        self.M = torch.clamp(self.M + delta_m, min=-self.max_memory, max=self.max_memory)

        # z = z + Σσ(K) / batch
        delta_z = sigma_k.sum(dim=(0, 1)) / batch

        # Clamp z to ensure valid normalization
        self.z = torch.clamp(self.z + delta_z, min=self.eps, max=self.max_norm)

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
        return f"dim={self._dim}, eps={self.eps}"
