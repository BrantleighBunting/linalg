# AGENTS.md

Guidance for AI coding assistants working with this repository.

## Project Overview

An educational machine learning project with modular transformer components built on linear algebra primitives:

1. **ai_comps/** - Modular transformer building blocks (pure NumPy, manual backprop)
2. **linalg/** - Pure-NumPy linear algebra toolkit
3. **gpt.py** - Decoder-only transformer training/inference

## Directory Structure

```
.
├── ai_comps/                   # Modular transformer components
│   ├── __init__.py                 # Package exports
│   ├── activations.py              # ReLU, GELU (planned: SwiGLU)
│   ├── normalization.py            # LayerNorm, RMSNorm
│   ├── positional.py               # Sinusoidal, Learned, RoPE
│   ├── attention.py                # Scaled dot-product, MHA
│   ├── tokenizers.py               # CharTokenizer (planned: BPE)
│   ├── cache.py                    # KVCache for inference
│   └── transformer.py              # Full encoder/decoder blocks
├── linalg/                     # Linear algebra primitives
│   ├── __init__.py                 # Public API exports
│   ├── qr.py                       # QR decomposition
│   ├── svd.py                      # Singular value decomposition
│   ├── eigen.py                    # Eigenvalue methods
│   ├── elimination.py              # Gaussian elimination
│   └── ...
├── gpt.py                      # GPT training script with AdamW
├── checkpoints_np/             # Pre-trained model weights (LFS)
├── tests/                      # pytest suite for linalg
└── pyproject.toml              # Project configuration
```

## Development Guidelines

### Dependencies

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'
```

### Running Tests

```bash
python -m pytest -q
python -m pytest tests/ -v
```

### Code Style

- **Black** for formatting (line-length 88)
- **isort** for imports (black profile)
- **mypy** for type checking (strict on `linalg/`)

## Key Patterns

### Modular Components (ai_comps/)

Components are designed to be swappable. Each module provides:
- A base implementation with forward/backward methods
- Factory functions like `get_activation()`, `get_norm()`, `get_positional_encoding()`
- Planned alternatives documented in module docstrings

```python
# Example: Swapping normalization
from ai_comps import get_norm
norm = get_norm("rmsnorm", d_model=512)  # or "layernorm"
```

### Transformer Architecture

- **Pre-LayerNorm**: LN before sublayer, not after
- All layers implement `forward()` returning output and `backward(dout)` returning gradients
- Shapes: `(B, T, D)` = batch, sequence, model dimension
- Multi-head attention: `(B, h, T, d)` where `D = h * d`

### Component Interfaces

**Normalization** (`normalization.py`):
```python
class LayerNorm:
    def forward(self, x) -> y
    def backward(self, dy) -> dx
    def step(self, lr, weight_decay)
```

**Attention** (`attention.py`):
```python
class MultiHeadAttention:
    def forward(self, X, mask=None, KV=None) -> Y
    def backward(self, dY) -> (dX, dKV)
    def step(self, lr, weight_decay)
```

**Positional** (`positional.py`):
```python
sinusoidal_encoding(max_len, d_model) -> np.ndarray
RotaryPositionalEmbedding.forward(q, k) -> (q_rot, k_rot)
```

### KV Cache for Inference

```python
from ai_comps import KVCache, LayerKVCache

cache = LayerKVCache(n_layers=4, batch_size=1, n_heads=4, max_seq_len=256, d_head=128)
k_full, v_full = cache[layer_idx].update(k_new, v_new)
```

## Extending Components

When adding new component variants:

1. Add implementation to the appropriate module
2. Register in factory function (e.g., `ACTIVATIONS` dict)
3. Update module docstring with "Planned" → "Implemented"
4. Add tests if applicable

## Important Notes

- The `linalg/` library is educational - prefer NumPy/SciPy for production
- All backpropagation is manual (no autograd) for learning purposes
- Checkpoint files use Git LFS (`checkpoints_np/*.npz`)
- GloVe embeddings in `data/` are licensed under ODC-PDDL 1.0
