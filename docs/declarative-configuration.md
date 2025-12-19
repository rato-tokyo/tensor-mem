# Declarative Configuration

## 定義

**Declarative Configuration**とは、モデルの構造がコード上で直接見える設計パターンです。

隠れたパラメータ（`num_layers=4`, `num_heads=8`など）で内部的にオブジェクトを生成するのではなく、実際のオブジェクト構造を呼び出し側で明示的に記述します。

## 使用例

```python
from tensor_mem import Layer, TensorMemory, TensorMemoryLM, default_memory_config

config = default_memory_config(dim=64)

model = TensorMemoryLM(
    vocab_size=32000,
    layers=[
        Layer([TensorMemory(config), TensorMemory(config), TensorMemory(config), TensorMemory(config)], hidden_size=256, d_ff=1024, bias=True, normalize_qkv=False),
        Layer([TensorMemory(config), TensorMemory(config), TensorMemory(config), TensorMemory(config)], hidden_size=256, d_ff=1024, bias=True, normalize_qkv=False),
        Layer([TensorMemory(config), TensorMemory(config), TensorMemory(config), TensorMemory(config)], hidden_size=256, d_ff=1024, bias=True, normalize_qkv=False),
        Layer([TensorMemory(config), TensorMemory(config), TensorMemory(config), TensorMemory(config)], hidden_size=256, d_ff=1024, bias=True, normalize_qkv=False),
    ],
)
```

この形式では：
- 4レイヤーであることが一目でわかる
- 各レイヤーに4つのメモリヘッドがあることが見える
- 各メモリの設定（dim=64）が明示的
- hidden_size, d_ffなどすべての設定が見える

## Anti-pattern（禁止）

```python
# BAD: ファクトリ関数で構造を隠す
model = small_model(vocab_size=32000, memory_type="standard")

# BAD: Configオブジェクトで構造を隠す
config = LMConfig(
    num_layers=4,      # 内部で4レイヤー生成（見えない）
    num_heads=4,       # 内部で4ヘッド生成（見えない）
    head_dim=64,
)
model = create_model(config)
```

この形式の問題：
- 実際に何が生成されるか見えない
- 設定と実体が分離している
- デバッグ時に構造を追いにくい

## 利点

1. **透明性**: 構造が一目でわかる
2. **一元化**: 設定と構造が同じ場所にある
3. **柔軟性**: 各レイヤー/ヘッドに異なる設定を適用可能
4. **デバッグ容易性**: 実際のオブジェクトを直接確認できる

## 関連パターン

- **Composite Pattern**: 木構造で全体と部分を同一視
- **Dependency Injection**: 事前構成済みインスタンスを注入
