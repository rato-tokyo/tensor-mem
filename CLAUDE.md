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

## Code Quality

- Python 3.11 specific
- Use `ruff` for linting
- Use `pytest` for testing
- All tests must pass before committing
