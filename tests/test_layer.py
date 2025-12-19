"""Tests for layer components."""

from __future__ import annotations

import torch
import torch.nn as nn

from tensor_mem import MemoryConfig
from tensor_mem.layer import FeedForwardLayer, PreNormBlock, TensorMemoryLayer


class TestFeedForwardLayer:
    """Tests for FeedForwardLayer."""

    def test_basic_init(self) -> None:
        """Test basic initialization."""
        ffn = FeedForwardLayer(hidden_size=256, d_ff=1024)
        assert isinstance(ffn.linear1, nn.Linear)
        assert isinstance(ffn.linear2, nn.Linear)
        assert isinstance(ffn.activation, nn.GELU)

    def test_shapes(self) -> None:
        """Test input/output shapes."""
        ffn = FeedForwardLayer(hidden_size=256, d_ff=1024)
        assert ffn.linear1.in_features == 256
        assert ffn.linear1.out_features == 1024
        assert ffn.linear2.in_features == 1024
        assert ffn.linear2.out_features == 256

    def test_forward(self) -> None:
        """Test forward pass."""
        ffn = FeedForwardLayer(hidden_size=256, d_ff=1024)
        x = torch.randn(2, 10, 256)
        out = ffn(x)
        assert out.shape == (2, 10, 256)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through FFN."""
        ffn = FeedForwardLayer(hidden_size=256, d_ff=1024)
        x = torch.randn(2, 10, 256, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestPreNormBlock:
    """Tests for PreNormBlock."""

    def test_basic_init(self) -> None:
        """Test basic initialization."""
        sublayer = nn.Linear(256, 256)
        block = PreNormBlock(hidden_size=256, sublayer=sublayer)
        assert isinstance(block.norm, nn.LayerNorm)
        assert block.sublayer is sublayer

    def test_forward_with_linear(self) -> None:
        """Test forward pass with linear sublayer."""
        sublayer = nn.Linear(256, 256)
        block = PreNormBlock(hidden_size=256, sublayer=sublayer)
        x = torch.randn(2, 10, 256)
        out = block(x)
        assert out.shape == (2, 10, 256)

    def test_residual_connection(self) -> None:
        """Test that residual connection is applied."""
        # Use zero weights to isolate residual
        sublayer = nn.Linear(256, 256)
        nn.init.zeros_(sublayer.weight)
        nn.init.zeros_(sublayer.bias)

        block = PreNormBlock(hidden_size=256, sublayer=sublayer)
        x = torch.randn(2, 10, 256)
        out = block(x)

        # With zero sublayer, output should equal input
        assert torch.allclose(out, x, atol=1e-6)

    def test_with_ffn(self) -> None:
        """Test PreNormBlock wrapping FeedForwardLayer."""
        ffn = FeedForwardLayer(hidden_size=256, d_ff=1024)
        block = PreNormBlock(hidden_size=256, sublayer=ffn)
        x = torch.randn(2, 10, 256)
        out = block(x)
        assert out.shape == (2, 10, 256)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through block."""
        ffn = FeedForwardLayer(hidden_size=256, d_ff=1024)
        block = PreNormBlock(hidden_size=256, sublayer=ffn)
        x = torch.randn(2, 10, 256, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestTensorMemoryLayer:
    """Tests for TensorMemoryLayer."""

    def test_basic_init(self) -> None:
        """Test basic initialization."""
        config = MemoryConfig(dim=64, eps=1e-6, use_delta_rule=False)
        layer = TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config)
        assert layer.attention.hidden_size == 256
        assert layer.attention.num_heads == 4

    def test_forward(self) -> None:
        """Test forward pass."""
        config = MemoryConfig(dim=64, eps=1e-6, use_delta_rule=False)
        layer = TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config)
        layer.reset_memory()
        x = torch.randn(2, 10, 256)
        out = layer(x)
        assert out.shape == (2, 10, 256)

    def test_gradient_flow(self) -> None:
        """Test gradient flows through layer."""
        config = MemoryConfig(dim=64, eps=1e-6, use_delta_rule=False)
        layer = TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config)
        layer.reset_memory()
        x = torch.randn(2, 10, 256, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_reset_memory(self) -> None:
        """Test memory reset."""
        config = MemoryConfig(dim=64, eps=1e-6, use_delta_rule=False)
        layer = TensorMemoryLayer(hidden_size=256, d_ff=1024, memory_config=config)
        layer.reset_memory()
        x = torch.randn(2, 10, 256)
        out1 = layer(x)

        # Reset and run again
        layer.reset_memory()
        out2 = layer(x)

        # Should be same after reset
        assert torch.allclose(out1, out2, atol=1e-6)
