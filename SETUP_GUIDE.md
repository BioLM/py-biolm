# BioLM Pipeline - Setup Guide

Complete setup guide for the BioLM Pipeline system using modern Python tooling.

---

## Prerequisites

- Python 3.9+ (3.10 recommended)
- Git
- `uv` (will be installed automatically)
- `direnv` (optional but recommended)

---

## Quick Start (TL;DR)

```bash
# Clone or navigate to repo
cd py-biolm

# Install everything
make install-all

# Activate environment (choose one)
direnv allow              # If you have direnv
source .venv/bin/activate # Otherwise

# Run tests
make test

# Try an example
make example-simple
```

---

## Detailed Setup

### 1. Install `uv` (Modern Python Package Manager)

`uv` is a fast Python package installer and resolver (similar to pip but much faster).

#### Option A: Automatic (via Makefile)
```bash
make install  # Will install uv if not present
```

#### Option B: Manual Install
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, you may need to restart your shell or run:
```bash
source ~/.bashrc  # or ~/.zshrc
```

Verify installation:
```bash
uv --version
```

---

### 2. Install `direnv` (Optional but Recommended)

`direnv` automatically activates the virtual environment when you enter the directory.

#### On Ubuntu/Debian:
```bash
sudo apt install direnv
```

#### On macOS:
```bash
brew install direnv
```

#### On other systems:
See: https://direnv.net/docs/installation.html

#### Configure your shell:

Add to `~/.bashrc` (or `~/.zshrc`):
```bash
eval "$(direnv hook bash)"  # For bash
# OR
eval "$(direnv hook zsh)"   # For zsh
```

Then restart your shell or run:
```bash
source ~/.bashrc
```

---

### 3. Install BioLM Pipeline

Navigate to the project directory:
```bash
cd /home/c/py-biolm
```

#### Option A: Full Installation (Recommended for Development)
```bash
make install-all
```

This installs:
- Core BioLM client
- Pipeline system
- All optional dependencies (UMAP, biotite, etc.)
- Development tools (black, ruff, mypy, pytest)

#### Option B: Core Only
```bash
make install
```

Installs just the core dependencies.

#### Option C: Core + Pipeline Extras
```bash
make install
make install-pipeline
```

---

### 4. Activate the Environment

#### Option A: With direnv (Automatic)
```bash
direnv allow
```

The environment will automatically activate whenever you enter the directory!

#### Option B: Manual Activation
```bash
source .venv/bin/activate
```

You'll see `(.venv)` in your prompt when activated.

---

## Verify Installation

### Check Python and packages:
```bash
python --version        # Should be 3.10 (or your chosen version)
python -c "import biolmai; print(biolmai.__version__)"
python -c "from biolmai.pipeline import DataPipeline; print('Pipeline OK')"
```

### Run tests:
```bash
make test
```

Expected output:
```
Test Summary
============================================================
Tests run: 105
Successes: 105
Failures: 0
Errors: 0
============================================================
```

---

## Environment Fix (NumPy Issue)

If you encounter NumPy version conflicts:

```bash
pip install 'numpy<2.0'
```

Or with uv:
```bash
uv pip install 'numpy<2.0'
```

---

## Project Structure

```
py-biolm/
├── .venv/                  # Virtual environment (auto-created)
├── .envrc                  # direnv configuration
├── .python-version         # Python version (3.10)
├── pyproject.toml          # Modern Python project config
├── Makefile                # Convenient commands
├── biolmai/                # Main package
│   ├── pipeline/           # Pipeline system
│   │   ├── datastore.py
│   │   ├── generative.py
│   │   ├── mlm_remasking.py
│   │   └── ...
│   └── ...
├── tests/                  # Test suite (105 tests)
│   ├── test_datastore.py
│   ├── test_pipeline.py
│   └── ...
└── examples/               # Usage examples
    ├── simple_pipeline_example.py
    └── advanced_pipeline_example.py
```

---

## Makefile Commands

### Installation
```bash
make install          # Core dependencies
make install-pipeline # + pipeline extras
make install-dev      # + dev tools
make install-all      # Everything (recommended)
make update           # Update dependencies
```

### Code Quality
```bash
make format           # Format with black
make lint             # Lint with ruff
make mypy             # Type check
make style            # format + lint
make check            # style + mypy
```

### Testing
```bash
make test             # Run all tests
make test-unit        # Unit tests only
make test-pipeline    # Pipeline tests
make test-pytest      # With pytest (if available)
```

### Examples
```bash
make example-simple   # Simple pipeline demo
make example-advanced # Advanced features demo
```

### Cleanup
```bash
make clean            # Remove build artifacts
make clean-all        # Remove everything including .venv
```

### Help
```bash
make help             # Show all commands
```

---

## Usage Examples

### Example 1: Quick Prediction

```python
from biolmai.pipeline import Predict

# One-liner prediction
df = Predict('temberture', sequences=['MKTAYIAKQRQ', 'MKLAVIDSAQ'])
print(df)
```

### Example 2: Multi-Stage Pipeline

```python
from biolmai.pipeline import DataPipeline, RankingFilter

pipeline = DataPipeline(sequences='sequences.csv')

# Parallel predictions
pipeline.add_predictions(['esmfold', 'temberture', 'proteinmpnn'])

# Ranking filter
pipeline.add_filter(RankingFilter('tm', n=100, ascending=False))

# Run
results = pipeline.run()
df = pipeline.get_final_data()
```

### Example 3: MLM Remasking

```python
from biolmai.pipeline import MLMRemasker, MODERATE_CONFIG

remasker = MLMRemasker(MODERATE_CONFIG)
variants = remasker.generate_variants('MKTAYIAKQRQ', num_variants=100)

for seq, metadata in variants:
    print(f"Variant: {seq}, Mutations: {metadata['num_mutations']}")
```

---

## Development Workflow

### 1. Make changes to code

### 2. Format and lint
```bash
make style
```

### 3. Run tests
```bash
make test
```

### 4. Type check
```bash
make mypy
```

### 5. Run all checks
```bash
make check
```

---

## Troubleshooting

### Issue: "direnv: command not found"
**Solution**: Install direnv or use manual activation:
```bash
source .venv/bin/activate
```

### Issue: "uv: command not found"
**Solution**: Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### Issue: NumPy version conflict
**Solution**: Downgrade NumPy:
```bash
uv pip install 'numpy<2.0'
```

### Issue: "Module not found" errors
**Solution**: Reinstall:
```bash
make clean-all
make install-all
```

### Issue: Tests fail
**Solution**: Check Python version and dependencies:
```bash
python --version  # Should be 3.9+
make update
make test
```

---

## Environment Variables

### BioLM API Key (Optional)

Set your API key in `~/.bashrc` or `~/.zshrc`:
```bash
export BIOLM_API_KEY="your-api-key-here"
```

Or create a `.env` file (not tracked by git):
```bash
echo 'BIOLM_API_KEY=your-api-key-here' > .env
```

---

## Updating the Project

### Update dependencies:
```bash
make update
```

### Pull latest changes:
```bash
git pull
make update
make test
```

---

## Uninstall

### Remove virtual environment:
```bash
make clean-all
```

### Remove direnv activation:
```bash
direnv revoke
```

---

## Comparison: Old vs New Setup

### Old Way (setup.py + pip)
```bash
pip install -e .
pip install pandas numpy matplotlib  # Manual deps
```

### New Way (pyproject.toml + uv)
```bash
make install-all  # Everything automated!
```

**Benefits**:
- ⚡ Much faster (uv is 10-100x faster than pip)
- 🔒 Better dependency resolution
- 🎯 Declarative dependencies in pyproject.toml
- 🛠️ Modern Python tooling
- 🔄 Easy to reproduce environments

---

## Additional Resources

### Documentation
- `README.rst` - Project overview
- `FINAL_SUMMARY.md` - Implementation details
- `PIPELINE_QUICKSTART.md` - Pipeline usage guide
- `tests/README.md` - Testing guide

### Examples
- `examples/simple_pipeline_example.py`
- `examples/advanced_pipeline_example.py`

### Tools
- uv: https://github.com/astral-sh/uv
- direnv: https://direnv.net/
- pyproject.toml: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

---

## Quick Reference

```bash
# Setup
make install-all && direnv allow

# Daily workflow
make style    # Before committing
make test     # After changes
make check    # Before PR

# Run code
python examples/simple_pipeline_example.py
python -m biolmai.pipeline

# Cleanup
make clean    # Artifacts only
make clean-all  # Everything
```

---

## Success! 🎉

You now have a fully configured BioLM Pipeline development environment with:
- ✅ Modern Python packaging (pyproject.toml)
- ✅ Fast package management (uv)
- ✅ Auto-activation (direnv)
- ✅ Quality tools (black, ruff, mypy)
- ✅ Comprehensive tests
- ✅ Working examples

**Happy coding!** 🚀
