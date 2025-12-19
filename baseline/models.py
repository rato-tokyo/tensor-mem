"""Baseline LLM models for comparison.

StandardTransformerLM: Traditional transformer with sinusoidal positional encoding.
TensorMemoryLM: Pure tensor product memory without positional encoding (NoPE).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tensor_mem import LinearMemoryAttention, MultiHeadMemory, TensorMemory


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class StandardTransformerBlock(nn.Module):
    """Standard transformer block with multi-head self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm architecture.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attn_mask: Optional causal mask

        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
        )
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TensorMemoryBlock(nn.Module):
    """Transformer block using tensor product memory attention.

    Uses Dependency Injection - receives a pre-configured LinearMemoryAttention.
    """

    def __init__(
        self,
        attention: LinearMemoryAttention,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = attention
        d_model = attention.hidden_size

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm architecture.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of same shape
        """
        # Memory attention with residual
        normed = self.norm1(x)
        attn_out = self.attention(normed)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x

    def reset_memory(self) -> None:
        """Reset the memory state."""
        self.attention.reset_memory()


class StandardTransformerLM(nn.Module):
    """Standard transformer language model with positional encoding.

    This serves as the baseline for comparison with tensor memory LLM.
    Uses sinusoidal positional encoding and standard softmax attention.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        """Initialize the standard transformer LM.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [StandardTransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            return_hidden: If True, also return hidden states

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
            Optionally also hidden states of shape (batch, seq_len, d_model)
        """
        # Embedding + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        hidden = self.norm(x)
        logits = self.lm_head(hidden)

        if return_hidden:
            return logits, hidden
        return logits

    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states without computing logits.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)

        Returns:
            Hidden states of shape (batch, seq_len, d_model)
        """
        _, hidden = self.forward(input_ids, return_hidden=True)
        return hidden


class TensorMemoryLM(nn.Module):
    """Tensor product memory language model without positional encoding (NoPE).

    This is the experimental model that uses only tensor product memory
    for sequence modeling, without any positional encoding.

    Uses Dependency Injection - receives pre-configured layer blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        layers: list[TensorMemoryBlock],
        dropout: float = 0.1,
    ):
        """Initialize the tensor memory LM.

        Args:
            vocab_size: Size of vocabulary
            layers: List of pre-configured TensorMemoryBlock instances
            dropout: Dropout rate for embedding
        """
        super().__init__()

        if not layers:
            raise ValueError("layers list cannot be empty")

        self.d_model = layers[0].attention.hidden_size

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(layers)

        self.norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            return_hidden: If True, also return hidden states

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
            Optionally also hidden states of shape (batch, seq_len, d_model)
        """
        # Embedding only (NO positional encoding)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.embed_dropout(x)

        # Tensor memory layers
        for layer in self.layers:
            x = layer(x)

        hidden = self.norm(x)
        logits = self.lm_head(hidden)

        if return_hidden:
            return logits, hidden
        return logits

    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states without computing logits.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)

        Returns:
            Hidden states of shape (batch, seq_len, d_model)
        """
        _, hidden = self.forward(input_ids, return_hidden=True)
        return hidden

    def reset_memory(self) -> None:
        """Reset memory state in all layers."""
        for layer in self.layers:
            layer.reset_memory()


def create_tensor_memory_lm(
    vocab_size: int,
    d_model: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    d_ff: int = 1024,
    dropout: float = 0.1,
    memory_eps: float = 1e-6,
    use_delta_rule: bool = False,
) -> TensorMemoryLM:
    """Factory function to create TensorMemoryLM with standard configuration.

    This is a convenience function that creates all the necessary components
    with Dependency Injection pattern.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        memory_eps: Epsilon for numerical stability in memory
        use_delta_rule: Whether to use Delta Rule for memory updates

    Returns:
        Configured TensorMemoryLM instance
    """
    head_dim = d_model // num_heads

    layers = []
    for _ in range(num_layers):
        # 1. Create memory instances
        memories = [
            TensorMemory(dim=head_dim, eps=memory_eps, use_delta_rule=use_delta_rule)
            for _ in range(num_heads)
        ]

        # 2. Create MultiHeadMemory
        mh_memory = MultiHeadMemory(memories)

        # 3. Create LinearMemoryAttention
        attention = LinearMemoryAttention(memory=mh_memory, hidden_size=d_model)

        # 4. Create TensorMemoryBlock
        block = TensorMemoryBlock(attention=attention, d_ff=d_ff, dropout=dropout)

        layers.append(block)

    return TensorMemoryLM(vocab_size=vocab_size, layers=layers, dropout=dropout)
