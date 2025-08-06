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

## Quick-start (with `uv`)

```bash
# 1. Clone + enter the repo
git clone https://github.com/BrantleighBunting/linalg.git
cd linalg

# 2. Create an isolated environment (Python ≥3.9)
uv venv .venv           # creates ./venv and installs uv’s pip shim
source .venv/bin/activate

# 3. Install runtime + dev dependencies
uv pip install -e '.[dev]'  # <-- the quotes avoid zsh globbing

# 4. Run the test-suite
python -m pytest -q
```
