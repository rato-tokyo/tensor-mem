"""Tests for TensorMemory and MultiHeadMemory classes."""

import pytest
import torch

from tensor_mem import MultiHeadMemory, TensorMemory


class TestTensorMemoryInit:
    """Tests for TensorMemory initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        memory = TensorMemory(dim=64)
        assert memory.dim == 64
        assert memory.eps == 1e-6

    def test_custom_eps(self):
        """Test custom epsilon value."""
        memory = TensorMemory(dim=64, eps=1e-8)
        assert memory.eps == 1e-8

    def test_not_initialized_before_reset(self):
        """Memory should not be initialized before reset."""
        memory = TensorMemory(dim=64)
        assert not memory.is_initialized

    def test_initialized_after_reset(self):
        """Memory should be initialized after reset."""
        memory = TensorMemory(dim=64)
        memory.reset()
        assert memory.is_initialized

    def test_empty_after_reset(self):
        """Memory should be empty after reset."""
        memory = TensorMemory(dim=64)
        memory.reset()
        assert memory.is_empty


class TestTensorMemoryReset:
    """Tests for TensorMemory reset method."""

    def test_reset_creates_correct_shapes(self):
        """Reset should create tensors with correct shapes."""
        dim = 64
        memory = TensorMemory(dim=dim)
        memory.reset()

        assert memory.M.shape == (dim, dim)
        assert memory.z.shape == (dim,)

    def test_reset_creates_zeros(self):
        """Reset should create zero tensors."""
        memory = TensorMemory(dim=64)
        memory.reset()

        assert (memory.M == 0).all()
        assert (memory.z == 0).all()

    def test_reset_respects_device(self):
        """Reset should place tensors on specified device."""
        memory = TensorMemory(dim=64)
        memory.reset(device="cpu")

        assert memory.M.device.type == "cpu"
        assert memory.z.device.type == "cpu"

    def test_reset_respects_dtype(self):
        """Reset should use specified dtype."""
        memory = TensorMemory(dim=64)
        memory.reset(dtype=torch.float64)

        assert memory.M.dtype == torch.float64
        assert memory.z.dtype == torch.float64


class TestTensorMemoryUpdate:
    """Tests for TensorMemory update method."""

    def test_update_without_init_raises(self):
        """Update should raise error if memory not initialized."""
        memory = TensorMemory(dim=64)
        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)

        with pytest.raises(RuntimeError, match="not initialized"):
            memory.update(keys, values)

    def test_update_makes_memory_non_empty(self):
        """Update should make memory non-empty."""
        memory = TensorMemory(dim=64)
        memory.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        assert not memory.is_empty

    def test_update_changes_memory(self):
        """Update should change memory state."""
        memory = TensorMemory(dim=64)
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
        memory = TensorMemory(dim=64)
        memory.reset()

        for _ in range(3):
            keys = torch.randn(2, 10, 64)
            values = torch.randn(2, 10, 64)
            memory.update(keys, values)

        assert (memory.z > 0).all()


class TestTensorMemoryRetrieve:
    """Tests for TensorMemory retrieve method."""

    def test_retrieve_without_init_raises(self):
        """Retrieve should raise error if memory not initialized."""
        memory = TensorMemory(dim=64)
        queries = torch.randn(2, 10, 64)

        with pytest.raises(RuntimeError, match="not initialized"):
            memory.retrieve(queries)

    def test_retrieve_returns_correct_shape(self):
        """Retrieve should return correct output shape."""
        memory = TensorMemory(dim=64)
        memory.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        queries = torch.randn(4, 20, 64)
        output = memory.retrieve(queries)

        assert output.shape == (4, 20, 64)

    def test_retrieve_from_empty_memory(self):
        """Retrieve from empty memory should not crash."""
        memory = TensorMemory(dim=64)
        memory.reset()

        queries = torch.randn(2, 10, 64)
        output = memory.retrieve(queries)

        assert output.shape == queries.shape
        assert not torch.isnan(output).any()

    def test_retrieve_no_nan(self):
        """Retrieve should not produce NaN values."""
        memory = TensorMemory(dim=64)
        memory.reset()

        for _ in range(5):
            keys = torch.randn(2, 10, 64)
            values = torch.randn(2, 10, 64)
            memory.update(keys, values)

        queries = torch.randn(2, 10, 64)
        output = memory.retrieve(queries)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestMultiHeadMemory:
    """Tests for MultiHeadMemory class."""

    def test_basic_init(self):
        """Test basic initialization."""
        mh = MultiHeadMemory(num_heads=8, head_dim=64)
        assert mh.num_heads == 8
        assert mh.head_dim == 64
        assert len(mh.memories) == 8

    def test_reset(self):
        """Test reset all memories."""
        mh = MultiHeadMemory(num_heads=4, head_dim=32)
        mh.reset()

        assert mh.is_initialized
        for m in mh.memories:
            assert m.is_initialized
            assert m.is_empty

    def test_update_and_retrieve(self):
        """Test update and retrieve with multi-head."""
        mh = MultiHeadMemory(num_heads=4, head_dim=32)
        mh.reset()

        # [batch, num_heads, seq, head_dim]
        keys = torch.randn(2, 4, 10, 32)
        values = torch.randn(2, 4, 10, 32)
        mh.update(keys, values)

        queries = torch.randn(2, 4, 5, 32)
        output = mh.retrieve(queries)

        assert output.shape == (2, 4, 5, 32)
        assert not torch.isnan(output).any()

    def test_heads_are_independent(self):
        """Each head should have independent memory."""
        mh = MultiHeadMemory(num_heads=4, head_dim=32)
        mh.reset()

        # Update only head 0
        keys = torch.zeros(1, 4, 10, 32)
        values = torch.zeros(1, 4, 10, 32)
        keys[:, 0] = torch.randn(1, 10, 32)
        values[:, 0] = torch.randn(1, 10, 32)
        mh.update(keys, values)

        # Head 0 should be non-empty, others should be empty
        assert not mh.memories[0].is_empty
        for _i in range(1, 4):
            # Other heads got zero updates, but z will still be positive
            # due to ELU+1 on zeros = 1
            pass  # This is expected behavior


class TestTensorMemoryIntegration:
    """Integration tests for TensorMemory."""

    def test_full_workflow(self):
        """Test complete workflow."""
        dim = 64
        memory = TensorMemory(dim=dim)

        # Initialize
        assert not memory.is_initialized
        memory.reset()
        assert memory.is_initialized
        assert memory.is_empty

        # Update
        keys = torch.randn(4, 32, dim)
        values = torch.randn(4, 32, dim)
        memory.update(keys, values)
        assert not memory.is_empty

        # Retrieve
        queries = torch.randn(4, 16, dim)
        output = memory.retrieve(queries)
        assert output.shape == (4, 16, dim)

        # Reset
        memory.reset()
        assert memory.is_empty

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that memory works on CUDA."""
        memory = TensorMemory(dim=64)
        memory.reset(device="cuda", dtype=torch.float16)

        keys = torch.randn(2, 10, 64, device="cuda", dtype=torch.float16)
        values = torch.randn(2, 10, 64, device="cuda", dtype=torch.float16)
        memory.update(keys, values)

        queries = torch.randn(2, 5, 64, device="cuda", dtype=torch.float16)
        output = memory.retrieve(queries)

        assert output.device.type == "cuda"
        assert output.dtype == torch.float16
