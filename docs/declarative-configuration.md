# Declarative Configuration Design Pattern

tensor-memプロジェクトで採用している設計パターンの詳細ドキュメント。

## 概要

**Declarative Configuration**は、モデル構造を明示的に宣言する設計パターンです。
隠れた`num_layers`や`num_heads`パラメータでオブジェクトを内部生成するのではなく、
実際のオブジェクト構造がコード上で直接見える形で構築します。

## モデル構造

### TensorMemoryLM の階層構造

```
TensorMemoryLM
├── embedding: Embedding(vocab_size, d_model)
├── embed_dropout: Dropout(dropout)
├── layers: ModuleList
│   ├── TensorMemoryBlock[0]
│   │   ├── attention: LinearMemoryAttention
│   │   │   ├── q_proj: Linear(hidden_size, proj_dim)
│   │   │   ├── k_proj: Linear(hidden_size, proj_dim)
│   │   │   ├── v_proj: Linear(hidden_size, proj_dim)
│   │   │   ├── o_proj: Linear(proj_dim, hidden_size)
│   │   │   └── memory: MultiHeadMemory
│   │   │       ├── TensorMemory[0] (dim=head_dim)
│   │   │       ├── TensorMemory[1] (dim=head_dim)
│   │   │       ├── TensorMemory[2] (dim=head_dim)
│   │   │       └── TensorMemory[3] (dim=head_dim)
│   │   ├── norm1: LayerNorm(d_model)
│   │   ├── norm2: LayerNorm(d_model)
│   │   └── ffn: Sequential
│   │       ├── Linear(d_model, d_ff)
│   │       ├── GELU()
│   │       ├── Dropout(dropout)
│   │       ├── Linear(d_ff, d_model)
│   │       └── Dropout(dropout)
│   ├── TensorMemoryBlock[1]
│   │   └── ... (same structure)
│   ├── TensorMemoryBlock[2]
│   │   └── ... (same structure)
│   └── TensorMemoryBlock[3]
│       └── ... (same structure)
├── norm: LayerNorm(d_model)
└── lm_head: Linear(d_model, vocab_size)  # weight tied with embedding
```

## プリセット構成

### small_model (約3Mパラメータ)

```
vocab_size: 任意
layers: 4
heads_per_layer: 4
head_dim: 64
hidden_size: 256 (= 4 heads × 64 dim)
d_ff: 1024
dropout: 0.1
```

### medium_model (約25Mパラメータ)

```
vocab_size: 任意
layers: 6
heads_per_layer: 8
head_dim: 64
hidden_size: 512 (= 8 heads × 64 dim)
d_ff: 2048
dropout: 0.1
```

### large_model (約110Mパラメータ)

```
vocab_size: 任意
layers: 12
heads_per_layer: 12
head_dim: 64
hidden_size: 768 (= 12 heads × 64 dim)
d_ff: 3072
dropout: 0.1
```

## 使用方法

### 基本的な使用

```python
from tensor_mem import small_model

# モデル作成（構造は内部で明示的に構築される）
model = small_model(vocab_size=32000, memory_type="standard")

# 推論
output = model(input_ids)

# メモリリセット（新しいシーケンスの前に）
model.reset_memory()
```

### カスタム構成

ファクトリ関数を使わず、直接構築することも可能：

```python
from tensor_mem import (
    TensorMemoryLM,
    TensorMemoryBlock,
    LinearMemoryAttention,
    MultiHeadMemory,
    TensorMemory,
    default_memory_config,
)

memory_config = default_memory_config(dim=64)

def make_layer():
    return TensorMemoryBlock(
        attention=LinearMemoryAttention(
            memory=MultiHeadMemory([
                TensorMemory(memory_config) for _ in range(4)
            ]),
            hidden_size=256,
            bias=True,
            normalize_qkv=False,
        ),
        d_ff=1024,
        dropout=0.1,
    )

model = TensorMemoryLM(
    vocab_size=32000,
    dropout=0.1,
    layers=[make_layer() for _ in range(4)],
)
```

## メモリタイプ

### standard (TensorMemory)

累積型メモリ。情報は永続的に蓄積される。

```python
model = small_model(vocab_size=32000, memory_type="standard")
```

### decaying (DecayingTensorMemory)

減衰型メモリ。古い情報は徐々に忘却される（decay=0.95）。

```python
model = small_model(vocab_size=32000, memory_type="decaying")
```

## 設計原則

### 1. 透明性

モデル構造がコード上で明示的に見える。

### 2. No Magic Numbers

`num_layers=4`のような隠れた設定ではなく、実際のオブジェクトが列挙される。

### 3. 柔軟性

必要に応じて各レイヤー/ヘッドに異なる設定を適用可能。

### 4. デバッグ容易性

作成されたオブジェクトを直接インスペクト可能。

## 関連パターン

- **Dependency Injection**: コンポーネントは設定ではなく、事前構成済みインスタンスを受け取る
- **No Default Arguments**: すべてのパラメータは明示的に指定される
