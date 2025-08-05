## Quick-start (Linux/macOS, Python ≥ 3.9)

```bash
# 1. Install Astral’s uv (one-shot static binary)
curl -LsSf https://astral.sh/uv/install.sh | sh
# make sure ~/.local/bin is on your PATH

# 2. Create an isolated virtual-environment in .venv/
uv venv .venv          # ⇢ python -m venv .venv, pip, uv all wired up

# 3. Activate it
source .venv/bin/activate  #  Windows: .venv\Scripts\activate

# 4. Install runtime + dev dependencies from pyproject.toml
uv pip install -e ".[dev]"

# 5. Run the test-suite
python -m pytest -q

```
