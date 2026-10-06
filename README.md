# llm-3v3-basketball-agents

multi-agent llm simulation for competitive 3v3 basketball (fiba rules).

## setup

### prerequisites
- python 3.11 or higher
- uv (recommended) or pip

### option 1: with uv (fast)

1. create virtual environment:
   ```bash
   uv venv
   ```

2. activate virtual environment:
   - windows (cmd / powershell):
     ```powershell
     .venv\Scripts\activate
     ```
   - linux / macos:
     ```bash
     source .venv/bin/activate
     ```

3. install packages and dev dependencies:
   ```bash
   uv sync --extra dev
   ```

### option 2: with standard venv and pip

1. create virtual environment:
   ```bash
   python -m venv .venv
   ```

2. activate virtual environment:
   - windows (cmd / powershell):
     ```powershell
     .venv\Scripts\activate
     ```
   - linux / macos:
     ```bash
     source .venv/bin/activate
     ```

3. install package in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```

## running tests and linter

run tests:
```bash
uv run pytest
# or if venv is activated: pytest
```

run linter:
```bash
uv run ruff check .
# or if venv is activated: ruff check .
```