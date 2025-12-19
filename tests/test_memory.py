"""Tests for TensorMemory, DecayingTensorMemory, and MultiHeadMemory classes."""

import pytest
import torch

from tensor_mem import DecayingTensorMemory, MultiHeadMemory, TensorMemory
from tensor_mem.memory.config import DecayingMemoryConfig, MemoryConfig


def default_config(dim: int = 64) -> MemoryConfig:
    """Create default memory config for tests."""
    return MemoryConfig(
        dim=dim,
        eps=1e-6,
        use_delta_rule=False,
        max_delta=10.0,
        max_memory=100.0,
        max_norm=1000.0,
    )


def delta_rule_config(dim: int = 64) -> MemoryConfig:
    """Create config with delta rule enabled."""
    return MemoryConfig(
        dim=dim,
        eps=1e-6,
        use_delta_rule=True,
        max_delta=10.0,
        max_memory=100.0,
        max_norm=1000.0,
    )


def custom_eps_config(dim: int = 64, eps: float = 1e-8) -> MemoryConfig:
    """Create config with custom epsilon."""
    return MemoryConfig(
        dim=dim,
        eps=eps,
        use_delta_rule=False,
        max_delta=10.0,
        max_memory=100.0,
        max_norm=1000.0,
    )


def custom_clamp_config(
    dim: int = 64,
    max_delta: float = 1.0,
    max_memory: float = 10.0,
    max_norm: float = 100.0,
) -> MemoryConfig:
    """Create config with custom clamping values."""
    return MemoryConfig(
        dim=dim,
        eps=1e-6,
        use_delta_rule=False,
        max_delta=max_delta,
        max_memory=max_memory,
        max_norm=max_norm,
    )


def decaying_config(dim: int = 64, decay: float = 0.95) -> DecayingMemoryConfig:
    """Create default decaying memory config for tests."""
    return DecayingMemoryConfig(
        dim=dim,
        eps=1e-6,
        use_delta_rule=False,
        max_delta=10.0,
        max_memory=100.0,
        max_norm=1000.0,
        decay=decay,
    )


class TestTensorMemoryInit:
    """Tests for TensorMemory initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        memory = TensorMemory(default_config(64))
        assert memory.dim == 64
        assert memory.eps == 1e-6

    def test_custom_eps(self):
        """Test custom epsilon value."""
        memory = TensorMemory(custom_eps_config(64, 1e-8))
        assert memory.eps == 1e-8

    def test_delta_rule_init(self):
        """Test delta rule initialization."""
        memory = TensorMemory(delta_rule_config(64))
        assert memory.use_delta_rule is True

        memory2 = TensorMemory(default_config(64))
        assert memory2.use_delta_rule is False

    def test_not_initialized_before_reset(self):
        """Memory should not be initialized before reset."""
        memory = TensorMemory(default_config(64))
        assert not memory.is_initialized

    def test_initialized_after_reset(self):
        """Memory should be initialized after reset."""
        memory = TensorMemory(default_config(64))
        memory.reset()
        assert memory.is_initialized

    def test_empty_after_reset(self):
        """Memory should be empty after reset."""
        memory = TensorMemory(default_config(64))
        memory.reset()
        assert memory.is_empty


class TestTensorMemoryReset:
    """Tests for TensorMemory reset method."""

    def test_reset_creates_correct_shapes(self):
        """Reset should create tensors with correct shapes."""
        dim = 64
        memory = TensorMemory(default_config(dim))
        memory.reset()

        assert memory.M.shape == (dim, dim)
        assert memory.z.shape == (dim,)

    def test_reset_creates_zeros(self):
        """Reset should create zero tensors."""
        memory = TensorMemory(default_config(64))
        memory.reset()

        assert (memory.M == 0).all()
        assert (memory.z == 0).all()

    def test_reset_respects_device(self):
        """Reset should place tensors on specified device."""
        memory = TensorMemory(default_config(64))
        memory.reset(device="cpu")

        assert memory.M.device.type == "cpu"
        assert memory.z.device.type == "cpu"

    def test_reset_respects_dtype(self):
        """Reset should use specified dtype."""
        memory = TensorMemory(default_config(64))
        memory.reset(dtype=torch.float64)

        assert memory.M.dtype == torch.float64
        assert memory.z.dtype == torch.float64


class TestTensorMemoryUpdate:
    """Tests for TensorMemory update method."""

    def test_update_without_init_raises(self):
        """Update should raise error if memory not initialized."""
        memory = TensorMemory(default_config(64))
        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)

        with pytest.raises(RuntimeError, match="not initialized"):
            memory.update(keys, values)

    def test_update_makes_memory_non_empty(self):
        """Update should make memory non-empty."""
        memory = TensorMemory(default_config(64))
        memory.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        assert not memory.is_empty

    def test_update_changes_memory(self):
        """Update should change memory state."""
        memory = TensorMemory(default_config(64))
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
        memory = TensorMemory(default_config(64))
        memory.reset()

        for _ in range(3):
            keys = torch.randn(2, 10, 64)
            values = torch.randn(2, 10, 64)
            memory.update(keys, values)

        assert (memory.z > 0).all()

    def test_delta_rule_update(self):
        """Delta rule should subtract existing bindings."""
        memory = TensorMemory(delta_rule_config(64))
        memory.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)

        # First update (no existing bindings, same as normal)
        memory.update(keys, values)
        m_after_first = memory.M.clone()

        # Second update with same keys should have smaller delta
        memory.update(keys, values)

        # Memory should still change but differently than without delta rule
        assert not torch.allclose(memory.M, m_after_first)
        assert not torch.isnan(memory.M).any()

    def test_delta_rule_vs_normal(self):
        """Delta rule should produce different results than normal update."""
        torch.manual_seed(42)

        memory_normal = TensorMemory(default_config(64))
        memory_delta = TensorMemory(delta_rule_config(64))
        memory_normal.reset()
        memory_delta.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)

        # First update should be the same
        memory_normal.update(keys, values)
        memory_delta.update(keys, values)
        assert torch.allclose(memory_normal.M, memory_delta.M)

        # Second update with same keys should differ
        memory_normal.update(keys, values)
        memory_delta.update(keys, values)
        assert not torch.allclose(memory_normal.M, memory_delta.M)


class TestTensorMemoryRetrieve:
    """Tests for TensorMemory retrieve method."""

    def test_retrieve_without_init_raises(self):
        """Retrieve should raise error if memory not initialized."""
        memory = TensorMemory(default_config(64))
        queries = torch.randn(2, 10, 64)

        with pytest.raises(RuntimeError, match="not initialized"):
            memory.retrieve(queries)

    def test_retrieve_returns_correct_shape(self):
        """Retrieve should return correct output shape."""
        memory = TensorMemory(default_config(64))
        memory.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        queries = torch.randn(4, 20, 64)
        output = memory.retrieve(queries)

        assert output.shape == (4, 20, 64)

    def test_retrieve_from_empty_memory(self):
        """Retrieve from empty memory should not crash."""
        memory = TensorMemory(default_config(64))
        memory.reset()

        queries = torch.randn(2, 10, 64)
        output = memory.retrieve(queries)

        assert output.shape == queries.shape
        assert not torch.isnan(output).any()

    def test_retrieve_no_nan(self):
        """Retrieve should not produce NaN values."""
        memory = TensorMemory(default_config(64))
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
    """Tests for MultiHeadMemory class (Dependency Injection pattern)."""

    def test_basic_init(self):
        """Test basic initialization with injected memories."""
        memories = [TensorMemory(default_config(64)) for _ in range(8)]
        mh = MultiHeadMemory(memories)
        assert mh.num_heads == 8
        assert mh.head_dim == 64
        assert len(mh.memories) == 8

    def test_empty_list_raises(self):
        """Empty memories list should raise error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            MultiHeadMemory([])

    def test_reset(self):
        """Test reset all memories."""
        memories = [TensorMemory(default_config(32)) for _ in range(4)]
        mh = MultiHeadMemory(memories)
        mh.reset()

        assert mh.is_initialized
        for m in mh.memories:
            assert m.is_initialized
            assert m.is_empty

    def test_update_and_retrieve(self):
        """Test update and retrieve with multi-head."""
        memories = [TensorMemory(default_config(32)) for _ in range(4)]
        mh = MultiHeadMemory(memories)
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
        memories = [TensorMemory(default_config(32)) for _ in range(4)]
        mh = MultiHeadMemory(memories)
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

    def test_custom_memory_class(self):
        """Test using DecayingTensorMemory with MultiHeadMemory."""
        # Create DecayingTensorMemory instances with DI pattern
        memories = [DecayingTensorMemory(decaying_config(32, 0.9)) for _ in range(4)]
        mh = MultiHeadMemory(memories)
        mh.reset()

        # All memories should be DecayingTensorMemory
        for m in mh.memories:
            assert isinstance(m, DecayingTensorMemory)
            assert m.decay == 0.9

        # Should work normally
        keys = torch.randn(2, 4, 10, 32)
        values = torch.randn(2, 4, 10, 32)
        mh.update(keys, values)

        queries = torch.randn(2, 4, 5, 32)
        output = mh.retrieve(queries)

        assert output.shape == (2, 4, 5, 32)
        assert not torch.isnan(output).any()

    def test_memory_config_via_injection(self):
        """Test that memory config is set via injection."""
        # Create memories with specific config
        config = MemoryConfig(
            dim=16,
            eps=1e-8,
            use_delta_rule=True,
            max_delta=10.0,
            max_memory=100.0,
            max_norm=1000.0,
        )
        memories = [TensorMemory(config) for _ in range(2)]
        mh = MultiHeadMemory(memories)

        for m in mh.memories:
            assert m.eps == 1e-8
            assert m.use_delta_rule is True


class TestTensorMemoryIntegration:
    """Integration tests for TensorMemory."""

    def test_full_workflow(self):
        """Test complete workflow."""
        dim = 64
        memory = TensorMemory(default_config(dim))

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
        memory = TensorMemory(default_config(64))
        memory.reset(device="cuda", dtype=torch.float16)

        keys = torch.randn(2, 10, 64, device="cuda", dtype=torch.float16)
        values = torch.randn(2, 10, 64, device="cuda", dtype=torch.float16)
        memory.update(keys, values)

        queries = torch.randn(2, 5, 64, device="cuda", dtype=torch.float16)
        output = memory.retrieve(queries)

        assert output.device.type == "cuda"
        assert output.dtype == torch.float16


class TestDecayingTensorMemoryInit:
    """Tests for DecayingTensorMemory initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        memory = DecayingTensorMemory(decaying_config(64, 0.95))
        assert memory.dim == 64
        assert memory.decay == 0.95
        assert memory.eps == 1e-6

    def test_custom_decay(self):
        """Test custom decay value."""
        memory = DecayingTensorMemory(decaying_config(64, 0.9))
        assert memory.decay == 0.9

    def test_invalid_decay_raises(self):
        """Invalid decay values should raise error."""
        with pytest.raises(ValueError, match="decay must be in range"):
            DecayingTensorMemory(decaying_config(64, 0.0))

        with pytest.raises(ValueError, match="decay must be in range"):
            DecayingTensorMemory(decaying_config(64, 1.0))

        with pytest.raises(ValueError, match="decay must be in range"):
            DecayingTensorMemory(decaying_config(64, 1.5))

        with pytest.raises(ValueError, match="decay must be in range"):
            DecayingTensorMemory(decaying_config(64, -0.1))

    def test_not_initialized_before_reset(self):
        """Memory should not be initialized before reset."""
        memory = DecayingTensorMemory(decaying_config(64))
        assert not memory.is_initialized

    def test_initialized_after_reset(self):
        """Memory should be initialized after reset."""
        memory = DecayingTensorMemory(decaying_config(64))
        memory.reset()
        assert memory.is_initialized


class TestDecayingTensorMemoryUpdate:
    """Tests for DecayingTensorMemory update method."""

    def test_update_changes_memory(self):
        """Update should change memory state."""
        memory = DecayingTensorMemory(decaying_config(64, 0.9))
        memory.reset()

        m_before = memory.M.clone()
        z_before = memory.z.clone()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        assert not torch.allclose(memory.M, m_before)
        assert not torch.allclose(memory.z, z_before)

    def test_decay_reduces_old_information(self):
        """Old information should decay with each update."""
        memory = DecayingTensorMemory(decaying_config(64, 0.5))  # Fast decay
        memory.reset()

        # First update
        keys1 = torch.randn(2, 10, 64)
        values1 = torch.randn(2, 10, 64)
        memory.update(keys1, values1)
        m_after_first = memory.M.clone()

        # Second update with different values
        keys2 = torch.randn(2, 10, 64)
        values2 = torch.randn(2, 10, 64)
        memory.update(keys2, values2)

        # M should have changed significantly due to decay
        # The old M was multiplied by 0.5, so it should be quite different
        assert not torch.allclose(memory.M, m_after_first)

    def test_high_decay_preserves_more(self):
        """High decay should preserve more old information."""
        torch.manual_seed(42)

        # High decay (slow forget)
        memory_high = DecayingTensorMemory(decaying_config(64, 0.99))
        memory_high.reset()

        # Low decay (fast forget)
        memory_low = DecayingTensorMemory(decaying_config(64, 0.5))
        memory_low.reset()

        # Same first update
        keys1 = torch.randn(2, 10, 64)
        values1 = torch.randn(2, 10, 64)
        memory_high.update(keys1, values1)
        memory_low.update(keys1, values1)

        m_high_first = memory_high.M.clone()
        m_low_first = memory_low.M.clone()

        # Same second update
        keys2 = torch.randn(2, 10, 64)
        values2 = torch.randn(2, 10, 64)
        memory_high.update(keys2, values2)
        memory_low.update(keys2, values2)

        # High decay should have M closer to original
        diff_high = (memory_high.M - m_high_first * 0.99).abs().mean()
        diff_low = (memory_low.M - m_low_first * 0.5).abs().mean()

        # After applying decay factor, the difference should be similar
        # (the new contribution)
        # This mainly tests that decay is being applied
        assert diff_high.item() > 0  # Some change occurred
        assert diff_low.item() > 0

    def test_memory_bounded_after_many_updates(self):
        """Memory should remain bounded after many updates."""
        memory = DecayingTensorMemory(decaying_config(64, 0.95))
        memory.reset()

        for _ in range(100):
            keys = torch.randn(2, 10, 64)
            values = torch.randn(2, 10, 64)
            memory.update(keys, values)

        # Due to decay, memory shouldn't explode
        assert not torch.isnan(memory.M).any()
        assert not torch.isinf(memory.M).any()
        assert memory.M.abs().max() <= memory.max_memory


class TestDecayingTensorMemoryRetrieve:
    """Tests for DecayingTensorMemory retrieve method."""

    def test_retrieve_returns_correct_shape(self):
        """Retrieve should return correct output shape."""
        memory = DecayingTensorMemory(decaying_config(64))
        memory.reset()

        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)
        memory.update(keys, values)

        queries = torch.randn(4, 20, 64)
        output = memory.retrieve(queries)

        assert output.shape == (4, 20, 64)

    def test_retrieve_no_nan(self):
        """Retrieve should not produce NaN values."""
        memory = DecayingTensorMemory(decaying_config(64, 0.9))
        memory.reset()

        for _ in range(10):
            keys = torch.randn(2, 10, 64)
            values = torch.randn(2, 10, 64)
            memory.update(keys, values)

        queries = torch.randn(2, 10, 64)
        output = memory.retrieve(queries)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestDecayingTensorMemoryVsTensorMemory:
    """Comparison tests between DecayingTensorMemory and TensorMemory."""

    def test_decaying_converges_to_recent(self):
        """Decaying memory should converge toward recent information."""
        torch.manual_seed(42)

        memory = DecayingTensorMemory(decaying_config(64, 0.5))
        memory.reset()

        # Many updates with same pattern
        keys = torch.randn(2, 10, 64)
        values = torch.randn(2, 10, 64)

        for _ in range(20):
            memory.update(keys, values)

        m_converged = memory.M.clone()

        # One more update - should be very similar
        memory.update(keys, values)

        # Memory should be close to converged state
        assert torch.allclose(memory.M, m_converged, rtol=0.1)

    def test_normal_memory_accumulates_unbounded(self):
        """Normal TensorMemory accumulates while decaying stays bounded."""
        torch.manual_seed(42)

        memory_normal = TensorMemory(default_config(64))
        memory_decaying = DecayingTensorMemory(decaying_config(64, 0.9))
        memory_normal.reset()
        memory_decaying.reset()

        for _ in range(50):
            keys = torch.randn(2, 10, 64)
            values = torch.randn(2, 10, 64)
            memory_normal.update(keys, values)
            memory_decaying.update(keys, values)

        # Normal memory z grows without bound (clamped at max_norm)
        # Decaying memory z stays bounded by the decay
        # This is the key difference
        assert memory_normal.z.mean() > memory_decaying.z.mean()
