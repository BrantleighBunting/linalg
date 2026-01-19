# AI Components

An educational machine learning project implementing neural network architectures
from scratch in pure NumPy. Features a modular transformer implementation with
manual backpropagation, built on a foundation of custom linear algebra primitives.

---

## Decoder-Only Transformer (GPT)

A character-level decoder-only transformer trained on TinyShakespeare, implemented
in pure NumPy with manual backpropagation.

### Run the pre-trained model

```bash
# Interactive REPL - type prompts, get Shakespeare-style completions
python gpt.py --repl

# With custom sampling parameters
python gpt.py --repl --temperature 0.8 --top_k 40 --gen_tokens 300
```

### Sampling options

| Flag | Description | Default |
|------|-------------|---------|
| `--temperature` | Lower = more deterministic | 1.0 |
| `--top_k` | Limit sampling to top-k tokens (0 = disabled) | 0 |
| `--gen_tokens` | Number of tokens to generate | 200 |

### Train from scratch

```bash
python gpt.py --train --steps 4000 --eval_every 200
```

**Model architecture:** 4 layers, 4 heads, 512-dim embeddings, 256 context length.

---

## Modular Components (`ai_comps/`)

Swappable building blocks for transformer experimentation. Part of an ongoing AI learning roadmap.

### Implemented

| Component | Paper |
|-----------|-------|
| Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) |
| LayerNorm | [Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba et al., 2016) |
| RMSNorm | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019) |
| GELU | [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) (Hendrycks & Gimpel, 2016) |
| RoPE | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021) |
| AdamW | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2017) |

### Roadmap (To Be Implemented)

| Component | Paper |
|-----------|-------|
| SwiGLU | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020) |
| GQA | [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023) |
| ALiBi | [Train Short, Test Long: Attention with Linear Biases](https://arxiv.org/abs/2108.12409) (Press et al., 2021) |
| BPE | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2015) |
| Flash Attention | [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) |

### Module Overview

| Module | Current | Planned |
|--------|---------|---------|
| `activations.py` | ReLU, GELU | SwiGLU, GeGLU |
| `normalization.py` | LayerNorm, RMSNorm | - |
| `positional.py` | Sinusoidal, Learned, RoPE | ALiBi |
| `attention.py` | Scaled Dot-Product, MHA | GQA, MQA, Flash |
| `tokenizers.py` | Character-level | BPE |
| `cache.py` | KV Cache | - |
| `transformer.py` | Encoder, Decoder, Full Transformer | - |

```python
from ai_comps import (
    LayerNorm, RMSNorm,           # Normalization
    gelu, get_activation,         # Activations
    RotaryPositionalEmbedding,    # Positional encodings
    MultiHeadAttention, KVCache,  # Attention + caching
    CharTokenizer,                # Tokenization
)
```

---

## Quick-start

### Option A: Using `uv` (recommended)

```bash
git clone https://github.com/BrantleighBunting/linalg.git
cd linalg

uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

uv pip install -e '.[dev]'
python -m pytest -q
```

### Option B: Using standard venv + pip

```bash
git clone https://github.com/BrantleighBunting/linalg.git
cd linalg

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e '.[dev]'
python -m pytest -q
```

---

## Linear Algebra Foundation (`linalg/`)

The project began as an educational linear algebra toolkit, providing the
mathematical primitives underlying neural network operations.

### Features

* **Decompositions**
  * Modified-Gram-Schmidt and Householder **QR**
  * Economy-size **SVD** via Strang's AᵀA eigen-route
* **Matrix utilities**
  * Determinant (`det`), adjugate (`adj`), matrix power (`matrix_power_eig`)
* **Linear systems & least-squares**
  * Gaussian elimination with partial pivoting (`gaussian_solve`)
  * Thin QR & Householder LS solvers
* **Iterative methods**
  * Power iteration (dominant eigenpair)
* **Rank / null-space tools**
  * Rank via elimination, null-space basis constructor

All public symbols are re-exported in `linalg/__init__.py`.

---

## Project Structure

```
.
├── ai_comps/           # Modular transformer components
│   ├── activations.py      # ReLU, GELU, etc.
│   ├── normalization.py    # LayerNorm, RMSNorm
│   ├── positional.py       # Sinusoidal, Learned, RoPE
│   ├── attention.py        # Scaled dot-product, MHA
│   ├── tokenizers.py       # Character-level tokenizer
│   ├── cache.py            # KV cache for inference
│   └── transformer.py      # Full transformer implementation
├── linalg/             # Linear algebra primitives
├── gpt.py              # GPT training/inference script
├── checkpoints_np/     # Pre-trained model weights
└── tests/              # Test suite
```
