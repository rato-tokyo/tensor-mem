# tensor-mem Development Guidelines

## Core Architecture - CRITICAL

### No Local Attention Design

**This project intentionally does NOT use local attention.**

Unlike the original Infini-attention paper which combines:
- Local attention (softmax-based, within current segment)
- Memory retrieval (from compressive memory)
- Learnable gate to blend both

tensor-mem uses **only tensor product memory**:
- **NO** local attention computation
- **NO** gate mechanism
- Pure linear attention through memory retrieval

### Correct Processing Flow

```
Forward pass:
1. Project input to Q, K, V
2. Apply σ (ELU+1) activation to Q and K
3. RETRIEVE from memory using σ(Q) → output
4. UPDATE memory with σ(K) and V
5. Return retrieved output (projected)
```

**CRITICAL**: The output comes from memory retrieval, NOT from computing attention over current sequence.

### Why This Design?

1. **Simplicity**: Single unified mechanism instead of hybrid
2. **True O(n) complexity**: No quadratic local attention
3. **Consistent behavior**: All information goes through the same memory path
4. **Infinite context**: Memory accumulates without segment boundaries

### Common Implementation Mistakes to Avoid

1. **Computing local attention and ignoring memory**: This defeats the purpose
2. **Updating memory but not using retrieve()**: Memory becomes write-only
3. **Adding gate mechanism**: Not part of this design
4. **Using softmax anywhere**: This is linear attention only

### Memory Operation Order

The order matters for causality:
1. **Retrieve FIRST** (using past memory state)
2. **Update AFTER** (add current KV to memory)

This ensures current tokens don't attend to themselves through memory.

## Dependency Injection Design Pattern

**All composable components use Dependency Injection.**

Instead of creating child objects internally with parameters passed through multiple layers,
components receive pre-configured instances from the outside.

### Correct Pattern

```python
# 1. Create memory instances with desired configuration
memories = [
    DecayingTensorMemory(dim=64, decay=0.95, eps=1e-6)
    for _ in range(num_heads)
]

# 2. Inject into MultiHeadMemory
multi_head = MultiHeadMemory(memories)

# 3. Inject into LinearMemoryAttention
attention = LinearMemoryAttention(memory=multi_head, hidden_size=512)

# 4. Inject into TensorMemoryBlock
block = TensorMemoryBlock(attention=attention, d_ff=2048)
```

### Benefits

1. **Single source of configuration**: All settings defined at instantiation point
2. **No parameter drilling**: Classes don't need to pass through kwargs they don't use
3. **Testability**: Easy to inject mocks for testing
4. **Flexibility**: Can mix different memory types per head if needed

### Anti-pattern (Do NOT do this)

```python
# BAD: Parameters drilled through multiple layers
class MultiHeadMemory:
    def __init__(self, num_heads, head_dim, memory_class, **kwargs):
        # Creates instances internally - hard to configure
        self.memories = [memory_class(dim=head_dim, **kwargs) for ...]
```

## Code Quality

- Python 3.11 specific
- Use `ruff` for linting
- Use `pytest` for testing
- All tests must pass before committing
