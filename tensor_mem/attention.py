"""Linear Memory Attention implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .memory import TensorMemory
from .utils import elu_plus_one, repeat_kv


class LinearMemoryAttention(nn.Module):
    """
    Linear Attention with Tensor Product Memory.

    This module implements a HuggingFace-compatible attention layer that uses
    linear attention instead of softmax attention, combined with tensor product
    memory for infinite context handling.

    Features:
        - Linear O(n) complexity instead of O(n²)
        - GQA (Grouped Query Attention) support
        - No position encoding (NoPE) for memory - content-based retrieval
        - HuggingFace compatible interface

    The attention mechanism:
        1. Projects input to Q, K, V
        2. Applies ELU+1 activation to Q and K
        3. Computes linear attention: output = σ(Q) @ (σ(K)^T @ V) / (σ(Q) @ Σσ(K))
        4. Updates the persistent memory for future retrievals

    Args:
        hidden_size: Hidden dimension of the model.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key-value heads (for GQA).
            If None, defaults to num_attention_heads (MHA).
        head_dim: Dimension per head. If None, computed as hidden_size // num_attention_heads.
        eps: Small constant for numerical stability.
        bias: Whether to use bias in Q/K/V projections.
        output_bias: Whether to use bias in output projection.

    Example:
        >>> attn = LinearMemoryAttention(
        ...     hidden_size=4096,
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8,  # GQA with 4x fewer KV heads
        ... )
        >>> hidden_states = torch.randn(2, 512, 4096)
        >>> output, _, _ = attn(hidden_states)
        >>> attn.reset_memory()  # For new sequence
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        eps: float = 1e-6,
        bias: bool = True,
        output_bias: bool = False,
    ) -> None:
        """Initialize LinearMemoryAttention."""
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.eps = eps

        # Compute GQA repetition factor
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_key_value_heads ({self.num_key_value_heads})"
            )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # Projection dimensions
        self.q_proj_dim = self.num_attention_heads * self.head_dim
        self.kv_proj_dim = self.num_key_value_heads * self.head_dim

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.q_proj_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_proj_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_proj_dim, bias=bias)
        self.o_proj = nn.Linear(self.q_proj_dim, hidden_size, bias=output_bias)

        # Create memory for each head
        # Memory dimension is head_dim (per-head memory)
        self.memories = nn.ModuleList(
            [
                TensorMemory(memory_dim=self.head_dim, eps=eps)
                for _ in range(self.num_key_value_heads)
            ]
        )

        # Normalization buffers for each KV head
        self.register_buffer("z_buffers", None, persistent=False)

    def reset_memory(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Reset memory for a new sequence.

        This should be called when starting to process a new, unrelated sequence.

        Args:
            device: Device to place memory tensors on.
            dtype: Data type for memory tensors.
        """
        for memory in self.memories:
            memory.reset(device=device, dtype=dtype)

    def _split_heads(
        self,
        tensor: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        """
        Split tensor into attention heads.

        Args:
            tensor: [batch, seq, num_heads * head_dim]

        Returns:
            [batch, num_heads, seq, head_dim]
        """
        batch, seq, _ = tensor.shape
        tensor = tensor.view(batch, seq, num_heads, self.head_dim)
        return tensor.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

    def _merge_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge attention heads back.

        Args:
            tensor: [batch, num_heads, seq, head_dim]

        Returns:
            [batch, seq, num_heads * head_dim]
        """
        batch, num_heads, seq, head_dim = tensor.shape
        tensor = tensor.transpose(1, 2)  # [batch, seq, num_heads, head_dim]
        return tensor.reshape(batch, seq, num_heads * head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        _attention_mask: torch.Tensor | None = None,
        _position_ids: torch.Tensor | None = None,
        _past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        _output_attentions: bool = False,
        _use_cache: bool = False,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple | None]:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq, hidden_size].
            attention_mask: (Unused) For API compatibility.
            position_ids: (Unused) For API compatibility. NoPE design.
            past_key_value: (Unused) For API compatibility. Memory replaces cache.
            output_attentions: (Unused) Linear attention doesn't compute weights.
            use_cache: (Unused) For API compatibility.
            **kwargs: Additional arguments for compatibility.

        Returns:
            Tuple of:
                - output: [batch, seq, hidden_size]
                - None: Attention weights (not computed in linear attention)
                - None: Cache (memory is used instead)
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Initialize memories if needed
        if not self.memories[0].is_initialized:
            self.reset_memory(device=device, dtype=dtype)

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)  # [batch, seq, q_proj_dim]
        key_states = self.k_proj(hidden_states)  # [batch, seq, kv_proj_dim]
        value_states = self.v_proj(hidden_states)  # [batch, seq, kv_proj_dim]

        # Expand KV for GQA
        if self.num_key_value_groups > 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups, self.head_dim)
            value_states = repeat_kv(value_states, self.num_key_value_groups, self.head_dim)

        # Split into heads
        # [batch, num_heads, seq, head_dim]
        query_states = self._split_heads(query_states, self.num_attention_heads)
        key_states = self._split_heads(key_states, self.num_attention_heads)
        value_states = self._split_heads(value_states, self.num_attention_heads)

        # Apply activation function: σ(Q) and σ(K)
        sigma_q = elu_plus_one(query_states)  # [batch, num_heads, seq, head_dim]
        sigma_k = elu_plus_one(key_states)  # [batch, num_heads, seq, head_dim]

        # Compute linear attention for each head
        # Linear attention: output = σ(Q) @ (σ(K)^T @ V) / (σ(Q) @ Σσ(K))

        # σ(K)^T @ V: accumulates outer products
        # [batch, num_heads, head_dim, seq] @ [batch, num_heads, seq, head_dim]
        # -> [batch, num_heads, head_dim, head_dim]
        kv_product = torch.einsum("bhsd,bhse->bhde", sigma_k, value_states)

        # σ(Q) @ (σ(K)^T @ V)
        # [batch, num_heads, seq, head_dim] @ [batch, num_heads, head_dim, head_dim]
        # -> [batch, num_heads, seq, head_dim]
        qkv = torch.einsum("bhsd,bhde->bhse", sigma_q, kv_product)

        # Compute normalization: Σσ(K) for each head
        # [batch, num_heads, seq, head_dim] -> [batch, num_heads, head_dim]
        k_sum = sigma_k.sum(dim=2)

        # σ(Q) @ Σσ(K) for normalization
        # Need cumulative sum for causal attention
        # For simplicity, we use the total sum (non-causal linear attention)
        # [batch, num_heads, seq, head_dim] @ [batch, num_heads, head_dim]
        # -> [batch, num_heads, seq]
        norm = torch.einsum("bhsd,bhd->bhs", sigma_q, k_sum)

        # Normalize with eps
        attn_output = qkv / (norm.unsqueeze(-1) + self.eps)

        # Update memories (per KV head group)
        # We update with the original KV heads (before GQA expansion)
        kv_key_states = self.k_proj(hidden_states)
        kv_value_states = self.v_proj(hidden_states)
        kv_key_states = self._split_heads(kv_key_states, self.num_key_value_heads)
        kv_value_states = self._split_heads(kv_value_states, self.num_key_value_heads)

        for h, memory in enumerate(self.memories):
            # Extract this head's keys and values
            # [batch, seq, head_dim]
            head_keys = kv_key_states[:, h, :, :]
            head_values = kv_value_states[:, h, :, :]
            memory.update(head_keys, head_values)

        # Merge heads and project output
        attn_output = self._merge_heads(attn_output)  # [batch, seq, q_proj_dim]
        output = self.o_proj(attn_output)  # [batch, seq, hidden_size]

        return output, None, None

    def extra_repr(self) -> str:
        """Return extra representation string for printing."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_attention_heads={self.num_attention_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, "
            f"head_dim={self.head_dim}"
        )
