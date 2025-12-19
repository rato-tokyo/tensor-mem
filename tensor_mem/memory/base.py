"""Base TensorMemory implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..utils import elu_plus_one
from .config import MemoryConfig


class BaseTensorMemory(nn.Module, ABC):
    """
    Abstract base class for Tensor Product Memory.

    Provides common functionality for all memory types:
    - Memory structure (M matrix and z vector)
    - Initialization and reset
    - Retrieval logic

    Subclasses must implement the `update` method.

    Memory structure:
        M: [dim, dim] - Associative matrix storing KV bindings
        z: [dim] - Normalization term (cumulative sum of keys)

    Retrieve formula:
        output = (σ(Q) @ M) / (σ(Q) @ z + eps)

    Where σ = ELU + 1 activation function.

    Uses config-based initialization - no default arguments.
    """

    def __init__(self, config: MemoryConfig) -> None:
        """Initialize BaseTensorMemory.

        Args:
            config: MemoryConfig containing all memory settings.
        """
        super().__init__()
        self._dim = config.dim
        self.eps = config.eps
        self.use_delta_rule = config.use_delta_rule
        self.max_delta = config.max_delta
        self.max_memory = config.max_memory
        self.max_norm = config.max_norm

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

    @abstractmethod
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

    def _retrieve_from_memory(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Core memory retrieval using activated queries/keys.

        This is the fundamental retrieval operation shared by both
        retrieve() and _compute_delta_values().

        Args:
            sigma: Activated tensor σ(Q) or σ(K) of shape [batch, seq, dim].

        Returns:
            Retrieved values of shape [batch, seq, dim].
        """
        # (σ @ M)
        retrieved = torch.matmul(sigma, self.M)

        # σ @ z
        norm = torch.matmul(sigma, self.z)

        # Normalize
        return retrieved / (norm.unsqueeze(-1) + self.eps)

    def _compute_update_matrices(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute update matrices for memory update.

        This is the common computation shared by TensorMemory and
        DecayingTensorMemory update methods.

        Args:
            keys: Key tensor of shape [batch, seq, dim].
            values: Value tensor of shape [batch, seq, dim].

        Returns:
            Tuple of (sigma_k, delta_m, delta_z):
                - sigma_k: Activated keys σ(K)
                - delta_m: Memory matrix update
                - delta_z: Normalization vector update
        """
        # Ensure memory is on the same device as keys
        if self.M.device != keys.device:
            self.M = self.M.to(keys.device)
            self.z = self.z.to(keys.device)

        batch, seq, _ = keys.shape

        sigma_k = elu_plus_one(keys)
        update_values = self._compute_delta_values(sigma_k, values)

        # M update: σ(K)^T @ V / (batch * seq)
        delta_m = torch.einsum("bsd,bse->de", sigma_k, update_values)
        delta_m = delta_m / (batch * seq)

        # Clamp delta to prevent overflow (especially in fp16)
        delta_m = torch.clamp(delta_m, min=-self.max_delta, max=self.max_delta)

        # z update: Σσ(K) / batch
        delta_z = sigma_k.sum(dim=(0, 1)) / batch

        return sigma_k, delta_m, delta_z

    def _clamp_memory(self) -> None:
        """
        Clamp memory values to prevent numerical instability.

        Ensures M stays within [-max_memory, max_memory] and
        z stays within [eps, max_norm].
        """
        self.M = torch.clamp(self.M, min=-self.max_memory, max=self.max_memory)
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

        # Ensure memory is on the same device as queries
        if self.M.device != queries.device:
            self.M = self.M.to(queries.device)
            self.z = self.z.to(queries.device)

        sigma_q = elu_plus_one(queries)
        return self._retrieve_from_memory(sigma_q)

    def _compute_delta_values(
        self,
        sigma_k: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute update values, applying Delta Rule if enabled.

        Args:
            sigma_k: Activated keys σ(K).
            values: Original values.

        Returns:
            Values to use for update (may be adjusted by Delta Rule).
        """
        if self.use_delta_rule and not self.is_empty:
            existing = self._retrieve_from_memory(sigma_k)
            return values - existing
        return values

    def _extra_repr_fields(self) -> list[str]:
        """Return list of field strings for extra_repr.

        Subclasses can override to add additional fields.
        """
        return [f"dim={self._dim}", f"eps={self.eps}"]

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return ", ".join(self._extra_repr_fields())
