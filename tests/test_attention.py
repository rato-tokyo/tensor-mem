"""Tests for LinearMemoryAttention class."""

import pytest
import torch

from tensor_mem import LinearMemoryAttention


class TestLinearMemoryAttentionInit:
    """Tests for LinearMemoryAttention initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        attn = LinearMemoryAttention(
            hidden_size=768,
            num_attention_heads=12,
        )
        assert attn.hidden_size == 768
        assert attn.num_attention_heads == 12
        assert attn.num_key_value_heads == 12  # Default to MHA
        assert attn.head_dim == 64  # 768 / 12

    def test_gqa_init(self):
        """Test GQA initialization."""
        attn = LinearMemoryAttention(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=4,
        )
        assert attn.num_attention_heads == 12
        assert attn.num_key_value_heads == 4
        assert attn.num_key_value_groups == 3

    def test_custom_head_dim(self):
        """Test custom head dimension."""
        attn = LinearMemoryAttention(
            hidden_size=768,
            num_attention_heads=12,
            head_dim=128,
        )
        assert attn.head_dim == 128

    def test_invalid_gqa_raises(self):
        """Invalid GQA configuration should raise error."""
        with pytest.raises(ValueError, match="must be divisible"):
            LinearMemoryAttention(
                hidden_size=768,
                num_attention_heads=12,
                num_key_value_heads=5,  # 12 not divisible by 5
            )

    def test_projection_shapes(self):
        """Check projection layer shapes."""
        attn = LinearMemoryAttention(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=4,
        )
        # Q projection: 768 -> 12 * 64 = 768
        assert attn.q_proj.in_features == 768
        assert attn.q_proj.out_features == 768

        # KV projection: 768 -> 4 * 64 = 256
        assert attn.k_proj.in_features == 768
        assert attn.k_proj.out_features == 256

        # Output projection: 768 -> 768
        assert attn.o_proj.in_features == 768
        assert attn.o_proj.out_features == 768

    def test_num_memories(self):
        """Number of memories should match KV heads."""
        attn = LinearMemoryAttention(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=4,
        )
        assert len(attn.memories) == 4


class TestLinearMemoryAttentionForward:
    """Tests for LinearMemoryAttention forward pass."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=4,
        )
        hidden_states = torch.randn(2, 32, 256)

        output, attn_weights, cache = attn(hidden_states)

        assert output.shape == (2, 32, 256)
        assert attn_weights is None  # Linear attention doesn't compute weights
        assert cache is None

    def test_gqa_forward(self):
        """Test forward with GQA."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
        )
        hidden_states = torch.randn(2, 32, 256)

        output, _, _ = attn(hidden_states)

        assert output.shape == (2, 32, 256)

    def test_variable_sequence_length(self):
        """Test with different sequence lengths."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=4,
        )

        for seq_len in [1, 16, 64, 256]:
            hidden_states = torch.randn(2, seq_len, 256)
            output, _, _ = attn(hidden_states)
            assert output.shape == (2, seq_len, 256)

    def test_gradient_flow(self):
        """Gradients should flow through forward."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=4,
        )
        hidden_states = torch.randn(2, 32, 256, requires_grad=True)

        output, _, _ = attn(hidden_states)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()

    def test_no_nan_outputs(self):
        """Output should not contain NaN values."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=4,
        )
        hidden_states = torch.randn(2, 32, 256)

        output, _, _ = attn(hidden_states)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_compatibility_args_ignored(self):
        """HuggingFace compatibility args should be accepted but ignored."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=4,
        )
        hidden_states = torch.randn(2, 32, 256)

        # These args should be accepted without error
        output, _, _ = attn(
            hidden_states,
            attention_mask=torch.ones(2, 32),
            position_ids=torch.arange(32).unsqueeze(0),
            past_key_value=None,
            output_attentions=True,
            use_cache=True,
        )

        assert output.shape == (2, 32, 256)


class TestLinearMemoryAttentionMemory:
    """Tests for memory functionality in LinearMemoryAttention."""

    def test_memory_reset(self):
        """Test memory reset."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=4,
        )
        hidden_states = torch.randn(2, 32, 256)

        # First forward initializes memories
        attn(hidden_states)

        # Check memories are initialized
        for memory in attn.memories:
            assert memory.is_initialized

        # Reset
        attn.reset_memory(device="cpu", dtype=torch.float32)

        # Check memories are reset
        for memory in attn.memories:
            assert memory.is_empty

    def test_memory_accumulates(self):
        """Memory should accumulate across forward passes."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=4,
        )
        hidden_states = torch.randn(2, 32, 256)

        # First forward
        attn(hidden_states)
        z_after_first = [m.z.clone() for m in attn.memories]

        # Second forward
        attn(hidden_states)
        z_after_second = [m.z.clone() for m in attn.memories]

        # z should have increased
        for z1, z2 in zip(z_after_first, z_after_second, strict=False):
            assert (z2 > z1).any() or (z2 != z1).any()

    def test_reset_clears_accumulation(self):
        """Reset should clear accumulated memory."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=4,
        )
        hidden_states = torch.randn(2, 32, 256)

        # Forward passes
        attn(hidden_states)
        attn(hidden_states)

        # Reset
        attn.reset_memory()

        # Check all memories are empty
        for memory in attn.memories:
            assert memory.is_empty


class TestLinearMemoryAttentionIntegration:
    """Integration tests for LinearMemoryAttention."""

    def test_transformer_block_pattern(self):
        """Test in a typical transformer block pattern."""
        hidden_size = 256
        num_heads = 4

        attn = LinearMemoryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )
        norm = torch.nn.LayerNorm(hidden_size)

        # Simulated residual connection
        x = torch.randn(2, 32, hidden_size)
        residual = x

        # Attention with pre-norm
        attn_out, _, _ = attn(norm(x))
        x = residual + attn_out

        assert x.shape == (2, 32, hidden_size)
        assert not torch.isnan(x).any()

    def test_multiple_layers(self):
        """Test with multiple attention layers."""
        hidden_size = 256
        num_layers = 4

        layers = torch.nn.ModuleList(
            [
                LinearMemoryAttention(
                    hidden_size=hidden_size,
                    num_attention_heads=4,
                )
                for _ in range(num_layers)
            ]
        )

        x = torch.randn(2, 32, hidden_size)

        for layer in layers:
            out, _, _ = layer(x)
            x = x + out

        assert x.shape == (2, 32, hidden_size)
        assert not torch.isnan(x).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that attention works on CUDA."""
        attn = LinearMemoryAttention(
            hidden_size=256,
            num_attention_heads=4,
        ).cuda()

        hidden_states = torch.randn(2, 32, 256, device="cuda")

        output, _, _ = attn(hidden_states)

        assert output.device.type == "cuda"
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision(self):
        """Test with mixed precision (float16)."""
        attn = (
            LinearMemoryAttention(
                hidden_size=256,
                num_attention_heads=4,
            )
            .cuda()
            .half()
        )

        hidden_states = torch.randn(2, 32, 256, device="cuda", dtype=torch.float16)

        output, _, _ = attn(hidden_states)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()
