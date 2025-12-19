"""Tensor Product Memory implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import elu_plus_one


class TensorMemory(nn.Module):
    """
    Tensor Product Memory (batch-shared).

    This class implements the core memory mechanism from Infini-attention,
    using associative matrices for key-value storage.

    The memory is shared across all batches (single M and z for all samples),
    which enables learning global patterns across the dataset.

    Memory structure:
        M: [memory_dim, memory_dim] - Associative matrix storing KV bindings
        z: [memory_dim] - Normalization term (cumulative sum of keys)

    Mathematical formulation:
        Update: M = M + σ(K)^T @ V / (batch * seq)
                z = z + Σσ(K) / batch
        Retrieve: output = (σ(Q) @ M) / (σ(Q) @ z + eps)

    Where σ = ELU + 1 activation function.

    Attributes:
        memory_dim: Dimension of the memory (typically hidden_size).
        eps: Small constant for numerical stability.
        M: Memory matrix buffer of shape [memory_dim, memory_dim].
        z: Normalization vector buffer of shape [memory_dim].

    Example:
        >>> memory = TensorMemory(memory_dim=768, eps=1e-6)
        >>> memory.reset(device="cuda", dtype=torch.float16)
        >>>
        >>> keys = torch.randn(4, 128, 768, device="cuda", dtype=torch.float16)
        >>> values = torch.randn(4, 128, 768, device="cuda", dtype=torch.float16)
        >>> memory.update(keys, values)
        >>>
        >>> queries = torch.randn(4, 32, 768, device="cuda", dtype=torch.float16)
        >>> output = memory.retrieve(queries)  # [4, 32, 768]
    """

    def __init__(
        self,
        memory_dim: int,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize TensorMemory.

        Args:
            memory_dim: Dimension of the memory vectors (typically hidden_size).
            eps: Small constant for numerical stability in normalization.
        """
        super().__init__()
        self._memory_dim = memory_dim
        self.eps = eps

        # Register buffers (persistent but not parameters)
        self.register_buffer("M", None, persistent=False)
        self.register_buffer("z", None, persistent=False)

    @property
    def is_initialized(self) -> bool:
        """Check if memory buffers have been initialized."""
        return self.M is not None and self.z is not None

    @property
    def is_empty(self) -> bool:
        """Check if memory is empty (all zeros or not initialized)."""
        if not self.is_initialized:
            return True
        return bool(torch.all(self.z == 0).item())

    @property
    def memory_dim(self) -> int:
        """Return the memory dimension."""
        return self._memory_dim

    def reset(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Reset memory to zeros.

        This should be called at the start of processing a new sequence
        or when you want to clear accumulated memory.

        Args:
            device: Device to place tensors on. If None, uses current device.
            dtype: Data type for tensors. If None, uses torch.float32.
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        self.M = torch.zeros(self._memory_dim, self._memory_dim, device=device, dtype=dtype)
        self.z = torch.zeros(self._memory_dim, device=device, dtype=dtype)

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Update memory with new key-value pairs.

        This performs the memory update step from Infini-attention:
            M = M + σ(K)^T @ V / (batch * seq)
            z = z + Σσ(K) / batch

        The normalization by batch size and sequence length ensures
        stable updates regardless of input size.

        Args:
            keys: Key tensor of shape [batch, seq, memory_dim].
            values: Value tensor of shape [batch, seq, memory_dim].

        Raises:
            RuntimeError: If memory has not been initialized with reset().
        """
        if not self.is_initialized:
            raise RuntimeError("Memory not initialized. Call reset(device, dtype) first.")

        batch, seq, _ = keys.shape

        # Apply activation function: σ(K) = ELU(K) + 1
        sigma_k = elu_plus_one(keys)  # [batch, seq, memory_dim]

        # Update memory matrix: M = M + σ(K)^T @ V / (batch * seq)
        # σ(K)^T @ V: [memory_dim, seq] @ [seq, memory_dim] -> [memory_dim, memory_dim]
        # We sum over batch dimension
        delta_m = torch.einsum("bsd,bse->de", sigma_k, values)
        self.M = self.M + delta_m / (batch * seq)

        # Update normalization: z = z + Σσ(K) / batch
        # Sum over batch and seq dimensions
        delta_z = sigma_k.sum(dim=(0, 1))  # [memory_dim]
        self.z = self.z + delta_z / batch

    def retrieve(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from memory based on queries.

        This performs the memory retrieval step from Infini-attention:
            output = (σ(Q) @ M) / (σ(Q) @ z + eps)

        Args:
            queries: Query tensor of shape [batch, seq, memory_dim].

        Returns:
            Retrieved values of shape [batch, seq, memory_dim].

        Raises:
            RuntimeError: If memory has not been initialized with reset().
        """
        if not self.is_initialized:
            raise RuntimeError("Memory not initialized. Call reset(device, dtype) first.")

        # Apply activation function: σ(Q) = ELU(Q) + 1
        sigma_q = elu_plus_one(queries)  # [batch, seq, memory_dim]

        # Retrieve: (σ(Q) @ M)
        # [batch, seq, memory_dim] @ [memory_dim, memory_dim] -> [batch, seq, memory_dim]
        retrieved = torch.matmul(sigma_q, self.M)

        # Compute normalization: σ(Q) @ z
        # [batch, seq, memory_dim] @ [memory_dim] -> [batch, seq]
        norm = torch.matmul(sigma_q, self.z)

        # Normalize with eps for numerical stability
        # Keep dimension for broadcasting: [batch, seq, 1]
        output = retrieved / (norm.unsqueeze(-1) + self.eps)

        return output

    def extra_repr(self) -> str:
        """Return extra representation string for printing."""
        return f"memory_dim={self._memory_dim}, eps={self.eps}"
