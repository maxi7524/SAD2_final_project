
# New

## Initialization 
To use project repository after fork you need to:
0. Install [`uv`](https://github.com/astral-sh/uv) (the fast Python package & environment manager by Astral)
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
1. Synchronize dependencies
```bash
uv sync
```
2. Activate environment
```bash
source .venv/bin/activate
```
3. Install library in development mode
```bash
uv pip install -e .
```
To run `path/script.py` from exercise 3, you need to run:
```bash
uv run python path/script
```


