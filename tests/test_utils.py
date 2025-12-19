"""Tests for utility functions."""

import pytest
import torch

from tensor_mem.utils import elu_plus_one


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
