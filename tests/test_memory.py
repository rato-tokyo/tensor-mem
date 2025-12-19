"""Tests for TensorMemory class."""

import pytest
import torch

from tensor_mem import TensorMemory


class TestTensorMemoryInit:
    """Tests for TensorMemory initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        memory = TensorMemory(memory_dim=768)
        assert memory.memory_dim == 768
        assert memory.eps == 1e-6

    def test_custom_eps(self):
        """Test custom epsilon value."""
        memory = TensorMemory(memory_dim=256, eps=1e-8)
        assert memory.eps == 1e-8

    def test_not_initialized_before_reset(self):
        """Memory should not be initialized before reset."""
        memory = TensorMemory(memory_dim=768)
        assert not memory.is_initialized

    def test_initialized_after_reset(self):
        """Memory should be initialized after reset."""
        memory = TensorMemory(memory_dim=768)
        memory.reset(device="cpu", dtype=torch.float32)
        assert memory.is_initialized

    def test_empty_after_reset(self):
        """Memory should be empty after reset."""
        memory = TensorMemory(memory_dim=768)
        memory.reset()
        assert memory.is_empty


class TestTensorMemoryReset:
    """Tests for TensorMemory reset method."""

    def test_reset_creates_correct_shapes(self):
        """Reset should create tensors with correct shapes."""
        dim = 256
        memory = TensorMemory(memory_dim=dim)
        memory.reset()

        assert memory.M.shape == (dim, dim)
        assert memory.z.shape == (dim,)

    def test_reset_creates_zeros(self):
        """Reset should create zero tensors."""
        memory = TensorMemory(memory_dim=128)
        memory.reset()

        assert (memory.M == 0).all()
        assert (memory.z == 0).all()

    def test_reset_respects_device(self):
        """Reset should place tensors on specified device."""
        memory = TensorMemory(memory_dim=64)
        memory.reset(device=torch.device("cpu"))

        assert memory.M.device.type == "cpu"
        assert memory.z.device.type == "cpu"

    def test_reset_respects_dtype(self):
        """Reset should use specified dtype."""
        memory = TensorMemory(memory_dim=64)
        memory.reset(dtype=torch.float64)

        assert memory.M.dtype == torch.float64
        assert memory.z.dtype == torch.float64

    def test_multiple_resets(self):
        """Multiple resets should work correctly."""
        memory = TensorMemory(memory_dim=64)

        for _ in range(3):
            memory.reset()
            assert memory.is_initialized
            assert memory.is_empty


class TestTensorMemoryUpdate:
    """Tests for TensorMemory update method."""

    def test_update_without_init_raises(self):
        """Update should raise error if memory not initialized."""
        memory = TensorMemory(memory_dim=64)
        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)

        with pytest.raises(RuntimeError, match="not initialized"):
            memory.update(keys, values)

    def test_update_makes_memory_non_empty(self):
        """Update should make memory non-empty."""
        memory = TensorMemory(memory_dim=64)
        memory.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        assert not memory.is_empty

    def test_update_changes_memory(self):
        """Update should change memory state."""
        memory = TensorMemory(memory_dim=64)
        memory.reset()

        m_before = memory.M.clone()
        z_before = memory.z.clone()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        assert not torch.allclose(memory.M, m_before)
        assert not torch.allclose(memory.z, z_before)

    def test_update_accumulates(self):
        """Multiple updates should accumulate."""
        memory = TensorMemory(memory_dim=64)
        memory.reset()

        for _i in range(3):
            keys = torch.randn(2, 10, 64)
            values = torch.randn(2, 10, 64)
            memory.update(keys, values)

        # z should have accumulated values
        assert (memory.z > 0).all()

    def test_update_gradient_flow(self):
        """Gradients should flow through update."""
        memory = TensorMemory(memory_dim=64)
        memory.reset()

        keys = torch.randn(2, 10, 64, requires_grad=True)
        values = torch.randn(2, 10, 64, requires_grad=True)
        memory.update(keys, values)

        # Compute loss using memory
        loss = memory.M.sum() + memory.z.sum()
        loss.backward()

        assert keys.grad is not None
        assert values.grad is not None


class TestTensorMemoryRetrieve:
    """Tests for TensorMemory retrieve method."""

    def test_retrieve_without_init_raises(self):
        """Retrieve should raise error if memory not initialized."""
        memory = TensorMemory(memory_dim=64)
        queries = torch.randn(2, 10, 64)

        with pytest.raises(RuntimeError, match="not initialized"):
            memory.retrieve(queries)

    def test_retrieve_returns_correct_shape(self):
        """Retrieve should return correct output shape."""
        memory = TensorMemory(memory_dim=64)
        memory.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        queries = torch.randn(4, 20, 64)
        output = memory.retrieve(queries)

        assert output.shape == (4, 20, 64)

    def test_retrieve_from_empty_memory(self):
        """Retrieve from empty memory should not crash."""
        memory = TensorMemory(memory_dim=64)
        memory.reset()

        queries = torch.randn(2, 10, 64)
        output = memory.retrieve(queries)

        # Output should be valid (though may be zeros or near-zeros)
        assert output.shape == queries.shape
        assert not torch.isnan(output).any()

    def test_retrieve_gradient_flow(self):
        """Gradients should flow through retrieve."""
        memory = TensorMemory(memory_dim=64)
        memory.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        queries = torch.randn(2, 5, 64, requires_grad=True)
        output = memory.retrieve(queries)
        loss = output.sum()
        loss.backward()

        assert queries.grad is not None

    def test_retrieve_after_multiple_updates(self):
        """Retrieve should work after multiple updates."""
        memory = TensorMemory(memory_dim=64)
        memory.reset()

        for _ in range(5):
            keys = torch.randn(2, 10, 64)
            values = torch.randn(2, 10, 64)
            memory.update(keys, values)

        queries = torch.randn(2, 10, 64)
        output = memory.retrieve(queries)

        assert output.shape == queries.shape
        assert not torch.isnan(output).any()


class TestTensorMemoryIntegration:
    """Integration tests for TensorMemory."""

    def test_full_workflow(self):
        """Test complete workflow: init -> reset -> update -> retrieve."""
        dim = 128
        batch, seq = 4, 32

        # Initialize
        memory = TensorMemory(memory_dim=dim)
        assert not memory.is_initialized

        # Reset
        memory.reset(dtype=torch.float32)
        assert memory.is_initialized
        assert memory.is_empty

        # Update
        keys = torch.randn(batch, seq, dim)
        values = torch.randn(batch, seq, dim)
        memory.update(keys, values)
        assert not memory.is_empty

        # Retrieve
        queries = torch.randn(batch, seq // 2, dim)
        output = memory.retrieve(queries)
        assert output.shape == (batch, seq // 2, dim)

        # Reset again
        memory.reset()
        assert memory.is_empty

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that memory works on CUDA."""
        memory = TensorMemory(memory_dim=64)
        memory.reset(device="cuda", dtype=torch.float16)

        keys = torch.randn(2, 10, 64, device="cuda", dtype=torch.float16)
        values = torch.randn(2, 10, 64, device="cuda", dtype=torch.float16)
        memory.update(keys, values)

        queries = torch.randn(2, 5, 64, device="cuda", dtype=torch.float16)
        output = memory.retrieve(queries)

        assert output.device.type == "cuda"
        assert output.dtype == torch.float16
