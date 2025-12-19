"""Tests for utility functions."""

import pytest
import torch

from tensor_mem.utils import elu_plus_one, repeat_kv


class TestEluPlusOne:
    """Tests for elu_plus_one activation function."""

    def test_output_positive(self):
        """All outputs should be positive."""
        x = torch.randn(100, 100)
        y = elu_plus_one(x)
        assert (y > 0).all(), "All outputs should be positive"

    def test_positive_input(self):
        """For positive inputs, output = x + 1."""
        x = torch.tensor([0.0, 1.0, 2.0, 10.0])
        y = elu_plus_one(x)
        expected = x + 1.0
        torch.testing.assert_close(y, expected)

    def test_negative_input(self):
        """For negative inputs, output = exp(x) - 1 + 1 = exp(x)."""
        x = torch.tensor([-1.0, -2.0, -10.0])
        y = elu_plus_one(x)
        # ELU(x) = exp(x) - 1 for x < 0, so ELU(x) + 1 = exp(x)
        expected = torch.exp(x)
        torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-5)

    def test_zero_input(self):
        """For zero input, output should be 1."""
        x = torch.tensor([0.0])
        y = elu_plus_one(x)
        assert y.item() == pytest.approx(1.0)

    def test_gradient_flow(self):
        """Gradients should flow through the function."""
        x = torch.randn(10, 10, requires_grad=True)
        y = elu_plus_one(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        shapes = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for shape in shapes:
            x = torch.randn(shape)
            y = elu_plus_one(x)
            assert y.shape == x.shape


class TestRepeatKv:
    """Tests for repeat_kv function."""

    def test_no_repeat(self):
        """When n_rep=1, output should equal input."""
        x = torch.randn(2, 128, 256)  # 4 heads, head_dim=64
        y = repeat_kv(x, n_rep=1, head_dim=64)
        torch.testing.assert_close(y, x)

    def test_repeat_2x(self):
        """Test 2x repetition (GQA with 2 query heads per KV head)."""
        batch, seq = 2, 128
        num_kv_heads, head_dim = 4, 64
        x = torch.randn(batch, seq, num_kv_heads * head_dim)

        y = repeat_kv(x, n_rep=2, head_dim=head_dim)

        expected_shape = (batch, seq, num_kv_heads * 2 * head_dim)
        assert y.shape == expected_shape

    def test_repeat_4x(self):
        """Test 4x repetition (GQA with 4 query heads per KV head)."""
        batch, seq = 2, 128
        num_kv_heads, head_dim = 2, 64
        x = torch.randn(batch, seq, num_kv_heads * head_dim)

        y = repeat_kv(x, n_rep=4, head_dim=head_dim)

        expected_shape = (batch, seq, num_kv_heads * 4 * head_dim)
        assert y.shape == expected_shape

    def test_values_repeated_correctly(self):
        """Verify values are repeated correctly."""
        batch, seq = 1, 2
        num_kv_heads, head_dim = 2, 4
        x = torch.arange(16).float().view(batch, seq, num_kv_heads * head_dim)

        y = repeat_kv(x, n_rep=2, head_dim=head_dim)

        # Shape should be [1, 2, 16]
        assert y.shape == (1, 2, 16)

        # First head [0,1,2,3] should appear twice, then second head [4,5,6,7] twice
        y_reshaped = y.view(batch, seq, num_kv_heads, 2, head_dim)
        for rep in range(2):
            for h in range(num_kv_heads):
                original = x.view(batch, seq, num_kv_heads, head_dim)[:, :, h, :]
                repeated = y_reshaped[:, :, h, rep, :]
                torch.testing.assert_close(repeated, original)

    def test_gradient_flow(self):
        """Gradients should flow through the function."""
        x = torch.randn(2, 128, 256, requires_grad=True)
        y = repeat_kv(x, n_rep=2, head_dim=64)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
