"""Tensor Product Memory implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import elu_plus_one


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

    For multi-head attention, create multiple TensorMemory instances.

    Args:
        dim: Dimension of the memory vectors.
        eps: Small constant for numerical stability.

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
    ) -> None:
        """Initialize TensorMemory."""
        super().__init__()
        self._dim = dim
        self.eps = eps

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
        return bool(torch.all(self.z == 0).item())

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

        # M = M + σ(K)^T @ V / (batch * seq)
        delta_m = torch.einsum("bsd,bse->de", sigma_k, values)
        self.M = self.M + delta_m / (batch * seq)

        # z = z + Σσ(K) / batch
        delta_z = sigma_k.sum(dim=(0, 1))
        self.z = self.z + delta_z / batch

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


class MultiHeadMemory(nn.Module):
    """
    Multi-head wrapper for TensorMemory.

    Creates multiple independent TensorMemory instances, one per head.
    This is a convenience wrapper for multi-head attention patterns.

    Args:
        num_heads: Number of memory heads.
        head_dim: Dimension per head.
        eps: Small constant for numerical stability.

    Example:
        >>> mh_memory = MultiHeadMemory(num_heads=8, head_dim=64)
        >>> mh_memory.reset(device="cuda")
        >>>
        >>> # keys/values: [batch, num_heads, seq, head_dim]
        >>> keys = torch.randn(4, 8, 128, 64, device="cuda")
        >>> values = torch.randn(4, 8, 128, 64, device="cuda")
        >>> mh_memory.update(keys, values)
        >>>
        >>> queries = torch.randn(4, 8, 32, 64, device="cuda")
        >>> output = mh_memory.retrieve(queries)  # [4, 8, 32, 64]
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
    ) -> None:
        """Initialize MultiHeadMemory."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.memories = nn.ModuleList(
            [TensorMemory(dim=head_dim, eps=eps) for _ in range(num_heads)]
        )

    @property
    def is_initialized(self) -> bool:
        """Check if all memories are initialized."""
        return all(m.is_initialized for m in self.memories)

    def reset(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Reset all memories."""
        for memory in self.memories:
            memory.reset(device=device, dtype=dtype)

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Update all memories.

        Args:
            keys: [batch, num_heads, seq, head_dim]
            values: [batch, num_heads, seq, head_dim]
        """
        for h, memory in enumerate(self.memories):
            memory.update(keys[:, h], values[:, h])

    def retrieve(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from all memories.

        Args:
            queries: [batch, num_heads, seq, head_dim]

        Returns:
            [batch, num_heads, seq, head_dim]
        """
        outputs = []
        for h, memory in enumerate(self.memories):
            out = memory.retrieve(queries[:, h])
            outputs.append(out)

        return torch.stack(outputs, dim=1)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}"
