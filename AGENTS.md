# AGENTS.md

Guidance for AI coding assistants working with this repository.

## Project Overview

This is an educational machine learning project with two main components:

1. **linalg/** - A pure-NumPy linear algebra toolkit implementing:
   - QR decomposition (Modified Gram-Schmidt, Householder)
   - SVD via eigenvalue decomposition
   - Gaussian elimination, least-squares solvers
   - Power iteration for dominant eigenpairs

2. **ai_comps/** - Transformer implementations:
   - `transformer.py` - Full encoder-decoder transformer in pure NumPy with manual backprop
   - `pytorch_encoder_decoder_transformer.py` - PyTorch version

3. **Root-level scripts** - Neural network examples:
   - `gpt.py` - GPT-2 style decoder training with AdamW optimizer
   - `xor.py`, `or.py` - Binary logic gate classifiers demonstrating backpropagation

## Directory Structure

```
.
├── linalg/                 # Pure-NumPy linear algebra library
│   ├── __init__.py         # Public API exports
│   ├── qr.py               # QR decomposition methods
│   ├── svd.py              # Singular value decomposition
│   ├── eigen.py            # Eigenvalue methods (power iteration)
│   ├── elimination.py      # Gaussian elimination, back-substitution
│   ├── matrix_functions.py # det, adj, rank
│   ├── projections.py      # Column space projection
│   └── utils.py            # Helper functions
├── ai_comps/               # Transformer implementations
│   ├── transformer.py      # NumPy transformer with full backprop
│   └── pytorch_encoder_decoder_transformer.py
├── tests/                  # pytest test suite for linalg
├── gpt.py                  # GPT training script
├── xor.py                  # XOR gate MLP example
├── or.py                   # OR gate MLP example
└── pyproject.toml          # Project configuration
```

## Development Guidelines

### Dependencies

This project uses `uv` for dependency management:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

### Running Tests

```bash
python -m pytest -q          # Run all tests
python -m pytest tests/ -v   # Verbose output
```

### Code Style

- **Black** for formatting (line-length 88)
- **isort** for import sorting (black profile)
- **mypy** for type checking (strict mode on `linalg/` package)

Run formatters:
```bash
black .
isort .
mypy linalg/
```

## Key Patterns

### NumPy Transformer (ai_comps/transformer.py)

- Uses **pre-LayerNorm** architecture (LN before sublayer, not after)
- All layers implement `forward()` returning `(output, cache)` and `backward(dout, cache)` returning `(dinput, grads)`
- Shapes follow convention: `(B, T, D)` = batch, sequence length, model dimension
- Multi-head attention uses `(B, h, T, d)` where `D = h * d`

### Linear Algebra Library (linalg/)

- Public API is re-exported from `linalg/__init__.py`
- Internal helpers stay in submodules
- Numerical stability is prioritized (e.g., pivoting in elimination)
- Tolerances scale with matrix dimensions via `scale_tol()`

### Neural Network Examples

- Use He initialization for ReLU networks
- Implement manual backward passes for educational purposes
- Cross-entropy loss with numerically stable softmax

## Important Notes

- The `linalg/` library is educational - prefer NumPy/SciPy for production use
- Transformer implementations prioritize clarity over performance
- All backpropagation is implemented manually (no autograd) for learning purposes
- GloVe embeddings in `data/` are licensed under ODC-PDDL 1.0
