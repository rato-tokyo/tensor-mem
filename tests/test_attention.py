"""Tests for LinearMemoryAttention class (Dependency Injection pattern)."""

import pytest
import torch

from tensor_mem import LinearMemoryAttention, MultiHeadMemory, TensorMemory
from tensor_mem.memory.config import MemoryConfig


def create_memory_config(
    head_dim: int = 64,
    eps: float = 1e-6,
    use_delta_rule: bool = False,
) -> MemoryConfig:
    """Create memory config for tests."""
    return MemoryConfig(
        dim=head_dim,
        eps=eps,
        use_delta_rule=use_delta_rule,
        max_delta=10.0,
        max_memory=100.0,
        max_norm=1000.0,
    )


def create_attention(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    bias: bool,
    normalize_qkv: bool,
    use_delta_rule: bool = False,
    eps: float = 1e-6,
) -> LinearMemoryAttention:
    """Helper to create LinearMemoryAttention with DI pattern."""
    config = create_memory_config(head_dim=head_dim, eps=eps, use_delta_rule=use_delta_rule)
    memories = [TensorMemory(config) for _ in range(num_heads)]
    mh_memory = MultiHeadMemory(memories)
    return LinearMemoryAttention(
        memory=mh_memory,
        hidden_size=hidden_size,
        bias=bias,
        normalize_qkv=normalize_qkv,
    )


class TestLinearMemoryAttentionInit:
    """Tests for LinearMemoryAttention initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
        assert attn.hidden_size == 256
        assert attn.num_heads == 4
        assert attn.head_dim == 64

    def test_custom_head_dim(self):
        """Test custom head dimension."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=32, bias=True, normalize_qkv=False
        )
        assert attn.head_dim == 32

    def test_projection_shapes(self):
        """Check projection layer shapes."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )

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
        attn = create_attention(
            hidden_size=256, num_heads=8, head_dim=32, bias=True, normalize_qkv=False
        )
        assert attn.memory.num_heads == 8
        assert len(attn.memory.memories) == 8

    def test_custom_head_dim_allows_any_num_heads(self):
        """Custom head_dim allows any num_heads regardless of hidden_size."""
        attn = create_attention(
            hidden_size=256, num_heads=3, head_dim=32, bias=True, normalize_qkv=False
        )
        assert attn.head_dim == 32
        assert attn.num_heads == 3

    def test_delta_rule_init(self):
        """Test delta rule initialization."""
        attn = create_attention(
            hidden_size=256,
            num_heads=4,
            head_dim=64,
            bias=True,
            normalize_qkv=False,
            use_delta_rule=True,
        )
        for m in attn.memory.memories:
            assert m.use_delta_rule is True

    def test_normalize_qkv_init(self):
        """Test normalize_qkv initialization."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=True
        )
        assert attn.normalize_qkv is True

        attn2 = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
        assert attn2.normalize_qkv is False


class TestLinearMemoryAttentionForward:
    """Tests for LinearMemoryAttention forward pass."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
        x = torch.randn(2, 32, 256)

        output = attn(x)

        assert output.shape == (2, 32, 256)

    def test_variable_sequence_length(self):
        """Test with different sequence lengths."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )

        for seq_len in [1, 16, 64, 256]:
            attn.reset_memory()
            x = torch.randn(2, seq_len, 256)
            output = attn(x)
            assert output.shape == (2, seq_len, 256)

    def test_gradient_flow(self):
        """Gradients should flow through forward."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
        x = torch.randn(2, 32, 256, requires_grad=True)

        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_outputs(self):
        """Output should not contain NaN values."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
        x = torch.randn(2, 32, 256)

        output = attn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestLinearMemoryAttentionMemory:
    """Tests for memory functionality in LinearMemoryAttention."""

    def test_memory_reset(self):
        """Test memory reset."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
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
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
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
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
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
        attn = create_attention(
            hidden_size=hidden_size, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
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
            [
                create_attention(
                    hidden_size=hidden_size,
                    num_heads=4,
                    head_dim=64,
                    bias=True,
                    normalize_qkv=False,
                )
                for _ in range(num_layers)
            ]
        )

        x = torch.randn(2, 32, hidden_size)

        for layer in layers:
            x = x + layer(x)

        assert x.shape == (2, 32, hidden_size)
        assert not torch.isnan(x).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that attention works on CUDA."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        ).cuda()
        x = torch.randn(2, 32, 256, device="cuda")

        output = attn(x)

        assert output.device.type == "cuda"
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision(self):
        """Test with mixed precision (float16)."""
        attn = (
            create_attention(
                hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
            )
            .cuda()
            .half()
        )
        x = torch.randn(2, 32, 256, device="cuda", dtype=torch.float16)

        output = attn(x)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()


class TestNumericalStability:
    """Tests for numerical stability features."""

    def test_normalize_qkv_forward(self):
        """Test forward with normalize_qkv enabled."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=True
        )
        x = torch.randn(2, 32, 256)

        output = attn(x)

        assert output.shape == (2, 32, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_values_without_normalization(self):
        """Large input values should be handled by clamping."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )
        # Large values that could cause overflow
        x = torch.randn(2, 32, 256) * 10

        output = attn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_values_with_normalization(self):
        """Large input values with normalization should be stable."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=True
        )
        x = torch.randn(2, 32, 256) * 10

        output = attn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_many_updates_stability(self):
        """Memory should remain stable after many updates."""
        attn = create_attention(
            hidden_size=256, num_heads=4, head_dim=64, bias=True, normalize_qkv=False
        )

        for _ in range(100):
            x = torch.randn(2, 32, 256)
            output = attn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Memory values should be bounded
        for m in attn.memory.memories:
            assert not torch.isnan(m.M).any()
            assert not torch.isinf(m.M).any()
            assert m.M.abs().max() <= m.max_memory

    def test_clamping_prevents_explosion(self):
        """Clamping should prevent memory explosion."""
        config = MemoryConfig(
            dim=64,
            eps=1e-6,
            use_delta_rule=False,
            max_delta=1.0,
            max_memory=10.0,
            max_norm=100.0,
        )
        memory = TensorMemory(config)
        memory.reset()

        # Repeatedly update with large values
        for _ in range(50):
            keys = torch.randn(2, 100, 64) * 5
            values = torch.randn(2, 100, 64) * 5
            memory.update(keys, values)

        # Memory should be bounded
        assert memory.M.abs().max() <= memory.max_memory
        assert memory.z.max() <= memory.max_norm
        assert not torch.isnan(memory.M).any()
        assert not torch.isinf(memory.M).any()
