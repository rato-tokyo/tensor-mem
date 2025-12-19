"""Linear Memory Attention implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import MultiHeadMemory
from .utils import elu_plus_one


class LinearMemoryAttention(nn.Module):
    """
    Linear Attention with Tensor Product Memory.

    A simple multi-head attention layer that uses ONLY tensor product memory.
    No local attention, no GQA - just straightforward memory-based retrieval.

    Uses Dependency Injection - receives a pre-configured MultiHeadMemory instance.

    Design (Causal Linear Attention):
        1. Project input to Q, K, V
        2. (Optional) L2 normalize Q, K, V for numerical stability
        3. Split into heads
        4. Apply σ (ELU+1) activation to Q and K
        5. Compute causal attention using cumulative sums:
           - cumsum_M[t] = Σ(i≤t) σ(K[i])^T @ V[i]
           - cumsum_z[t] = Σ(i≤t) σ(K[i])
           - output[t] = (σ(Q[t]) @ cumsum_M[t]) / (σ(Q[t]) @ cumsum_z[t] + eps)
        6. UPDATE persistent memory with final cumsum values
        7. Merge heads and project output

    The persistent memory (MultiHeadMemory) accumulates across sequences,
    while causal attention is computed within each sequence.

    Args:
        memory: Pre-configured MultiHeadMemory instance.
        hidden_size: Hidden dimension of the model.
        bias: Whether to use bias in projections.
        normalize_qkv: Whether to L2 normalize Q, K, V after projection.
            Recommended for fp16 training to prevent overflow.

    Example:
        >>> from tensor_mem import TensorMemory, MultiHeadMemory
        >>>
        >>> # 1. Create memory instances
        >>> memories = [TensorMemory(dim=64, eps=1e-6) for _ in range(8)]
        >>>
        >>> # 2. Create MultiHeadMemory
        >>> mh_memory = MultiHeadMemory(memories)
        >>>
        >>> # 3. Inject into LinearMemoryAttention
        >>> attn = LinearMemoryAttention(
        ...     memory=mh_memory,
        ...     hidden_size=512,
        ...     bias=True,
        ...     normalize_qkv=False,
        ... )
        >>>
        >>> x = torch.randn(2, 128, 512)
        >>> output = attn(x)
        >>> attn.reset_memory()  # For new sequence
    """

    def __init__(
        self,
        memory: MultiHeadMemory,
        hidden_size: int,
        bias: bool,
        normalize_qkv: bool,
    ) -> None:
        """Initialize LinearMemoryAttention."""
        super().__init__()

        self.hidden_size = hidden_size
        self.memory = memory
        self.num_heads = memory.num_heads
        self.head_dim = memory.head_dim
        self.normalize_qkv = normalize_qkv
        self.eps = memory.memories[0].eps

        proj_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, proj_dim, bias=bias)
        self.o_proj = nn.Linear(proj_dim, hidden_size, bias=False)

    def reset_memory(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Reset memory for a new sequence."""
        self.memory.reset(device=device, dtype=dtype)

    def _causal_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal linear attention using cumulative sums.

        Args:
            q: Activated queries σ(Q), shape [batch, num_heads, seq, head_dim]
            k: Activated keys σ(K), shape [batch, num_heads, seq, head_dim]
            v: Values, shape [batch, num_heads, seq, head_dim]

        Returns:
            Output tensor [batch, num_heads, seq, head_dim]
        """
        batch, num_heads, seq_len, head_dim = q.shape

        # Compute KV outer products: [batch, num_heads, seq, head_dim, head_dim]
        # kv[b, h, t, d1, d2] = k[b, h, t, d1] * v[b, h, t, d2]
        kv = torch.einsum("bhsd,bhse->bhsde", k, v)

        # Cumulative sum for causal attention
        # cumsum_kv[t] = Σ(i≤t) K[i]^T @ V[i]
        cumsum_kv = torch.cumsum(kv, dim=2)  # [batch, num_heads, seq, head_dim, head_dim]

        # Cumulative sum for normalization
        # cumsum_k[t] = Σ(i≤t) K[i]
        cumsum_k = torch.cumsum(k, dim=2)  # [batch, num_heads, seq, head_dim]

        # Add persistent memory contribution
        if self.memory.is_initialized:
            for h in range(num_heads):
                mem = self.memory.memories[h]
                if mem.M.device != q.device:
                    mem.M = mem.M.to(q.device)
                    mem.z = mem.z.to(q.device)
                # Add persistent memory to cumsum
                cumsum_kv[:, h] = cumsum_kv[:, h] + mem.M.unsqueeze(0).unsqueeze(0)
                cumsum_k[:, h] = cumsum_k[:, h] + mem.z.unsqueeze(0).unsqueeze(0)

        # Compute attention output: Q @ cumsum_KV
        # output[b, h, t, d] = Σ_d1 q[b, h, t, d1] * cumsum_kv[b, h, t, d1, d]
        numerator = torch.einsum("bhsd,bhsde->bhse", q, cumsum_kv)

        # Compute normalization: Q @ cumsum_K
        # norm[b, h, t] = Σ_d q[b, h, t, d] * cumsum_k[b, h, t, d]
        denominator = torch.einsum("bhsd,bhsd->bhs", q, cumsum_k)

        # Normalize
        output: torch.Tensor = numerator / (denominator.unsqueeze(-1) + self.eps)

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq, hidden_size].

        Returns:
            Output tensor [batch, seq, hidden_size].
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Initialize memory if needed
        if not self.memory.is_initialized:
            self.reset_memory(device=device, dtype=dtype)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Optional L2 normalization for numerical stability (recommended for fp16)
        if self.normalize_qkv:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            v = F.normalize(v, p=2, dim=-1)

        # Split into heads: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply σ (ELU+1) activation to Q and K
        sigma_q = elu_plus_one(q)
        sigma_k = elu_plus_one(k)

        # Compute causal linear attention
        output = self._causal_linear_attention(sigma_q, sigma_k, v)

        # Update persistent memory with current sequence's KV
        self.memory.update(sigma_k, v)

        # Merge heads: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
        output = output.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

        # Project output
        result: torch.Tensor = self.o_proj(output)

        return result

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}"
        )
