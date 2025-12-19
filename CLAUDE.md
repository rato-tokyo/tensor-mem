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

## Declarative Configuration - IMMUTABLE RULE

**モデル構造は呼び出し側で直接見える形で記述する。**

このルールは削除・変更不可。ファクトリ関数でモデルを生成することは禁止。

### 正しい形式

```python
model = TensorMemoryLM(
    vocab_size=32000,
    layers=[
        Layer([TensorMemory(config), TensorMemory(config), TensorMemory(config), TensorMemory(config)]),
        Layer([TensorMemory(config), TensorMemory(config), TensorMemory(config), TensorMemory(config)]),
        Layer([TensorMemory(config), TensorMemory(config), TensorMemory(config), TensorMemory(config)]),
        Layer([TensorMemory(config), TensorMemory(config), TensorMemory(config), TensorMemory(config)]),
    ],
)
```

この形式では：
- 4レイヤーであることが一目でわかる
- 各レイヤーに4つのメモリヘッドがあることが見える
- 構造と設定が一元化されている

### Why This Design?

1. **透明性**: 構造が一目でわかる
2. **一元化**: 設定と構造が同じ場所にある
3. **柔軟性**: 各レイヤー/ヘッドに異なる設定を適用可能
4. **デバッグ容易性**: 実際のオブジェクトを直接確認できる

### Anti-pattern (NEVER do this)

```python
# BAD: ファクトリ関数で構造を隠す
model = small_model(vocab_size=32000, memory_type="standard")

# BAD: Configオブジェクトで構造を隠す
config = LMConfig(num_layers=4, num_heads=8, ...)
model = create_model(config)
```

ファクトリ関数やConfigオブジェクトは構造を隠蔽するため禁止。

## Dependency Injection

**Components receive pre-configured instances, not configuration.**

This works together with Declarative Configuration:

```python
# 1. Create memory instances
memories = [TensorMemory(config) for _ in range(num_heads)]

# 2. Inject into MultiHeadMemory
multi_head = MultiHeadMemory(memories)

# 3. Inject into LinearMemoryAttention
attention = LinearMemoryAttention(
    memory=multi_head,
    hidden_size=512,
    bias=True,
    normalize_qkv=False,
)

# 4. Inject into TensorMemoryBlock
block = TensorMemoryBlock(attention=attention, d_ff=2048, dropout=0.1)
```

### Benefits

1. **Single source of configuration**: All settings defined at instantiation point
2. **No parameter drilling**: Classes don't need to pass through kwargs they don't use
3. **Testability**: Easy to inject mocks for testing
4. **Flexibility**: Can mix different memory types per head if needed

## No Legacy Code Policy

**Keeping old code for backward compatibility is strictly prohibited.**

When refactoring or changing APIs:
1. **Delete old implementations completely** - Do not keep deprecated methods/classes
2. **No compatibility shims** - Do not add wrappers to support old APIs
3. **No `# deprecated` comments** - Remove code instead of marking it
4. **Update all usages** - Fix all call sites when changing interfaces
5. **Clean tests** - Update tests to use new APIs, don't test old ones

### Why?

- Old code creates confusion about which API to use
- Maintenance burden increases with duplicate code
- Tests become unreliable when testing deprecated paths
- New contributors may accidentally use old patterns

### When Changing APIs

```python
# WRONG: Adding backward compatibility
def old_method(self):  # deprecated
    return self.new_method()

# CORRECT: Remove old_method entirely, update all callers
```

## No Default Arguments Policy - IMMUTABLE RULE

**Default arguments in function/method signatures are strictly prohibited.**

This rule cannot be removed or modified. All parameters must be explicitly passed.

### Why?

1. **Explicit is better than implicit** - Callers must consciously choose all values
2. **No hidden configuration** - All settings visible at call site
3. **Prevents accidental misconfiguration** - No "I forgot to set X" bugs
4. **Forces centralized config** - Settings defined in one place, not scattered in defaults

### Correct Pattern

```python
# Configuration dataclass (single source of truth)
@dataclass
class MemoryConfig:
    dim: int
    eps: float
    use_delta_rule: bool
    max_delta: float
    max_memory: float
    max_norm: float

# Class with no defaults - requires config
class TensorMemory:
    def __init__(self, config: MemoryConfig) -> None:
        self.dim = config.dim
        self.eps = config.eps
        # ...
```

### Anti-pattern (NEVER do this)

```python
# BAD: Hidden defaults
def __init__(self, dim: int, eps: float = 1e-6, use_delta_rule: bool = False):
    ...

# BAD: Optional with default None
def __init__(self, config: Config | None = None):
    if config is None:
        config = Config()  # Hidden default!
```

### Exception

Only `None` is allowed as default for truly optional parameters that change behavior:
```python
def reset(self, device: torch.device | None = None) -> None:
    # None means "use current device" - not a hidden configuration
```

## Code Quality

- Python 3.11 specific
- Use `ruff` for linting
- Use `pytest` for testing
- All tests must pass before committing
