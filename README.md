# linalg

An educational, **pure-NumPy** linear-algebra toolkit that grows hand-in-hand
with a “Transformer-from-scratch” study project.
---

## Features

* **Decompositions**
  * Modified-Gram–Schmidt and Householder **QR**
  * Economy-size **SVD** via Strang’s \(A^{\mathsf T}A\) eigen-route
* **Matrix utilities**
  * Determinant (`det`), adjugate (`adj`)
  * Matrix Power  `matrix_power_eig`
* **Linear systems & least-squares**
  * Gaussian elimination with partial pivoting (`gaussian_solve`)
  * Thin QR & Householder LS solvers
* **Iterative methods**
  * Power iteration (dominant eigenpair)
* **Rank / null-space tools**
  * Rank via elimination, null-space basis constructor
* **Data**
  * GloVe 6B-300d embeddings (ODC-PDDL 1.0)

All public symbols are re-exported in `linalg/__init__.py`; internal helpers
remain in sub-modules to keep the surface small.

---

## Quick-start

### Option A: Using `uv` (recommended)

```bash
# 1. Clone + enter the repo
git clone https://github.com/BrantleighBunting/linalg.git
cd linalg

# 2. Create an isolated environment (Python >=3.10)
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install runtime + dev dependencies
uv pip install -e '.[dev]'  # quotes avoid zsh globbing

# 4. Run the test-suite
python -m pytest -q
```

### Option B: Using standard venv + pip

```bash
# 1. Clone + enter the repo
git clone https://github.com/BrantleighBunting/linalg.git
cd linalg

# 2. Create virtual environment (Python >=3.10)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e '.[dev]'

# 4. Run the test-suite
python -m pytest -q
```

---

## GPT Model

A character-level GPT-2 style decoder trained on TinyShakespeare, implemented in pure NumPy with manual backpropagation.

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

Model architecture: 4 layers, 4 heads, 512-dim embeddings, 256 context length.
