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

# 4. Inject into Layer
layer = Layer(memories, hidden_size=512, d_ff=2048, bias=True, normalize_qkv=False)
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

## No Dropout Policy - IMMUTABLE RULE

**Dropout is permanently prohibited in this project.**

This rule cannot be removed or modified. Dropout layers must never be added.

### Why?

1. **Simplicity**: Fewer hyperparameters to tune
2. **Inference consistency**: Training and inference behave identically
3. **Memory mechanism**: Tensor product memory provides its own regularization through information compression
4. **Reproducibility**: Same input always produces same output

### Prohibited

```python
# NEVER do this
nn.Dropout(0.1)
nn.Dropout(p=0.5)
F.dropout(x, p=0.1)
```

### Alternative Regularization

If regularization is needed, use:
- Weight decay in optimizer
- Gradient clipping
- Early stopping
- Data augmentation

## No CLI Parameters Policy - IMMUTABLE RULE

**CLIパラメータでの設定指定は厳禁。**

このルールは削除・変更不可。すべての設定はconfigファイルで管理する。

### Why?

1. **再現性**: 実験設定がコードとして残る
2. **透明性**: 設定がファイルに明示的に記述される
3. **バージョン管理**: 設定変更がgit履歴に残る
4. **Declarative Configurationとの一貫性**: 構造も設定もコードで明示

### Correct Pattern

```python
# scripts/config.py
@dataclass(frozen=True)
class ExperimentConfig:
    device: str = "cuda"
    d_model: int = 256
    num_heads: int = 4
    # ...

EXPERIMENT_CONFIG = ExperimentConfig()

# scripts/compare.py
def main() -> None:
    cfg = EXPERIMENT_CONFIG
    # Use cfg.d_model, cfg.num_heads, etc.
```

### Anti-pattern (NEVER do this)

```python
# BAD: argparseでパラメータ受け取り
parser = argparse.ArgumentParser()
parser.add_argument("--d-model", type=int, default=256)
parser.add_argument("--num-layers", type=int, default=4)
args = parser.parse_args()

# BAD: forループでモデル構造を動的生成
layers = [Layer(...) for _ in range(args.num_layers)]
```

CLIパラメータは設定を隠蔽し、Declarative Configurationの原則に違反する。

## Global Variables for Configuration - IMMUTABLE RULE

**設定とモデル定義はすべてグローバル変数で直接定義する。**

このルールは削除・変更不可。クラスや関数でラップしない。

### Correct Pattern (scripts/models.py)

```python
# 設定: 単純なグローバル変数
DEVICE = "cuda"
D_MODEL = 256
NUM_HEADS = 4
D_FF = 1024
VOCAB_SIZE = 10000
HEAD_DIM = D_MODEL // NUM_HEADS

MAX_EPOCHS = 50
PATIENCE = 2
SEQ_LEN = 64
BATCH_SIZE = 32
LR = 1e-3

# メモリ設定: グローバル変数
MEMORY_CONFIG = MemoryConfig(
    dim=HEAD_DIM,
    eps=1e-6,
    use_delta_rule=False,
)

# モデル定義: グローバル変数（Declarative Configuration）
TENSOR_MEMORY_MODEL = TensorMemoryLM(
    vocab_size=VOCAB_SIZE,
    layers=[
        Layer([TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), ...], ...),
        Layer([TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), ...], ...),
        Layer([TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), ...], ...),
        Layer([TensorMemory(MEMORY_CONFIG), TensorMemory(MEMORY_CONFIG), ...], ...),
    ],
)
```

### Anti-pattern (NEVER do this)

```python
# BAD: dataclassで設定をラップ
@dataclass
class ExperimentConfig:
    d_model: int = 256
    ...

# BAD: 関数でモデル生成
def create_model(vocab_size: int) -> TensorMemoryLM:
    ...

# BAD: クラスで設定を管理
class Config:
    def __init__(self):
        self.d_model = 256
```

クラスや関数は不要な抽象化。グローバル変数で直接定義すれば十分。

## Code Quality

- Python 3.11 specific
- Use `ruff` for linting
- Use `pytest` for testing
- All tests must pass before committing
