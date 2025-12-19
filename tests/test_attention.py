"""Tests for LinearMemoryAttention class."""

import pytest
import torch

from tensor_mem import LinearMemoryAttention


class TestLinearMemoryAttentionInit:
    """Tests for LinearMemoryAttention initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4)
        assert attn.hidden_size == 256
        assert attn.num_heads == 4
        assert attn.head_dim == 64  # 256 / 4

    def test_custom_head_dim(self):
        """Test custom head dimension."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4, head_dim=32)
        assert attn.head_dim == 32

    def test_projection_shapes(self):
        """Check projection layer shapes."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4)

        assert attn.q_proj.in_features == 256
        assert attn.q_proj.out_features == 256
        assert attn.k_proj.in_features == 256
        assert attn.k_proj.out_features == 256
        assert attn.v_proj.in_features == 256
        assert attn.v_proj.out_features == 256
        assert attn.o_proj.in_features == 256
        assert attn.o_proj.out_features == 256

    def test_memory_structure(self):
        """Memory should match num_heads."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=8)
        assert attn.memory.num_heads == 8
        assert len(attn.memory.memories) == 8


class TestLinearMemoryAttentionForward:
    """Tests for LinearMemoryAttention forward pass."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4)
        x = torch.randn(2, 32, 256)

        output = attn(x)

        assert output.shape == (2, 32, 256)

    def test_variable_sequence_length(self):
        """Test with different sequence lengths."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4)

        for seq_len in [1, 16, 64, 256]:
            x = torch.randn(2, seq_len, 256)
            output = attn(x)
            assert output.shape == (2, seq_len, 256)

    def test_gradient_flow(self):
        """Gradients should flow through forward."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4)
        x = torch.randn(2, 32, 256, requires_grad=True)

        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_outputs(self):
        """Output should not contain NaN values."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4)
        x = torch.randn(2, 32, 256)

        output = attn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestLinearMemoryAttentionMemory:
    """Tests for memory functionality in LinearMemoryAttention."""

    def test_memory_reset(self):
        """Test memory reset."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4)
        x = torch.randn(2, 32, 256)

        # First forward initializes memories
        attn(x)
        assert attn.memory.is_initialized

        # Reset
        attn.reset_memory()
        for m in attn.memory.memories:
            assert m.is_empty

    def test_memory_accumulates(self):
        """Memory should accumulate across forward passes."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4)
        x = torch.randn(2, 32, 256)

        # First forward
        attn(x)
        z_after_first = [m.z.clone() for m in attn.memory.memories]

        # Second forward
        attn(x)
        z_after_second = [m.z.clone() for m in attn.memory.memories]

        # z should have increased
        for z1, z2 in zip(z_after_first, z_after_second, strict=False):
            assert not torch.allclose(z1, z2)

    def test_reset_clears_accumulation(self):
        """Reset should clear accumulated memory."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4)
        x = torch.randn(2, 32, 256)

        # Forward passes
        attn(x)
        attn(x)

        # Reset
        attn.reset_memory()

        for m in attn.memory.memories:
            assert m.is_empty


class TestLinearMemoryAttentionIntegration:
    """Integration tests for LinearMemoryAttention."""

    def test_transformer_block_pattern(self):
        """Test in a typical transformer block pattern."""
        hidden_size = 256
        attn = LinearMemoryAttention(hidden_size=hidden_size, num_heads=4)
        norm = torch.nn.LayerNorm(hidden_size)

        x = torch.randn(2, 32, hidden_size)
        residual = x

        # Attention with pre-norm
        attn_out = attn(norm(x))
        x = residual + attn_out

        assert x.shape == (2, 32, hidden_size)
        assert not torch.isnan(x).any()

    def test_multiple_layers(self):
        """Test with multiple attention layers."""
        hidden_size = 256
        num_layers = 4

        layers = torch.nn.ModuleList(
            [LinearMemoryAttention(hidden_size=hidden_size, num_heads=4) for _ in range(num_layers)]
        )

        x = torch.randn(2, 32, hidden_size)

        for layer in layers:
            x = x + layer(x)

        assert x.shape == (2, 32, hidden_size)
        assert not torch.isnan(x).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that attention works on CUDA."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4).cuda()
        x = torch.randn(2, 32, 256, device="cuda")

        output = attn(x)

        assert output.device.type == "cuda"
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision(self):
        """Test with mixed precision (float16)."""
        attn = LinearMemoryAttention(hidden_size=256, num_heads=4).cuda().half()
        x = torch.randn(2, 32, 256, device="cuda", dtype=torch.float16)

        output = attn(x)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()
