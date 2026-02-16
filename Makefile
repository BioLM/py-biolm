.PHONY: install install-dev install-all clean clean-build clean-pyc clean-test coverage dist docs docs-iframe help lint lint/flake8 style check format mypy test test-pipeline test-unit

.DEFAULT_GOAL := help

# Install core dependencies using uv
install:
	@echo "🔍 Checking for uv..."
	@if ! command -v uv &> /dev/null; then \
		echo "📥 Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "✅ uv installed. You may need to restart your shell or run: source ~/.bashrc"; \
	else \
		echo "✅ uv is already installed"; \
	fi
	@echo ""
	@echo "🚀 Creating virtual environment with Python $(shell cat .python-version)..."
	uv venv --python $(shell cat .python-version)
	@echo ""
	@echo "📦 Installing core dependencies..."
	uv pip install -e .
	@echo ""
	@echo "✅ Installation complete!"
	@echo ""
	@echo "🔧 To activate the environment, run one of:"
	@echo "   • direnv allow           (if you have direnv)"
	@echo "   • source .venv/bin/activate"
	@echo ""
	@echo "💡 For development mode with all extras, run: make install-all"

# Install with pipeline extras
install-pipeline:
	@echo "📦 Installing with pipeline extras..."
	uv pip install -e ".[pipeline]"
	@echo "✅ Pipeline extras installed!"

# Install development dependencies
install-dev:
	@echo "📦 Installing development dependencies..."
	uv pip install -e ".[dev]"
	@echo ""
	@echo "🔧 Setting up pre-commit hooks..."
	@if [ -d .venv ]; then \
		.venv/bin/pre-commit install --install-hooks 2>/dev/null || echo "⚠️  pre-commit install skipped (not available)"; \
	fi
	@echo "✅ Development dependencies installed!"

# Install everything (all extras + dev)
install-all: install
	@echo ""
	@echo "📦 Installing all extras..."
	uv pip install -e ".[all,dev]"
	@echo ""
	@echo "🔧 Setting up pre-commit hooks..."
	@if [ -d .venv ]; then \
		.venv/bin/pre-commit install --install-hooks 2>/dev/null || echo "⚠️  pre-commit install skipped (not available)"; \
	fi
	@echo ""
	@echo "✅ Full installation complete!"
	@echo ""
	@echo "🎉 Ready to use!"
	@echo "   • BioLM client is installed"
	@echo "   • Pipeline system is available"
	@echo "   • Development tools are ready"

# Update dependencies
update:
	@echo "🔄 Updating dependencies..."
	uv pip install --upgrade -e ".[all,dev]"
	@echo "✅ Dependencies updated!"

# Run code formatting with black
format:
	@echo "🎨 Formatting code with black..."
	@if [ -d .venv ]; then \
		.venv/bin/black biolmai tests examples; \
	else \
		black biolmai tests examples; \
	fi

# Run linting with ruff
lint:
	@echo "🔍 Linting code with ruff..."
	@if [ -d .venv ]; then \
		.venv/bin/ruff check biolmai tests examples; \
	else \
		ruff check biolmai tests examples; \
	fi

# Run type checking with mypy
mypy:
	@echo "🔬 Type checking with mypy..."
	@if [ -d .venv ]; then \
		.venv/bin/mypy biolmai; \
	else \
		mypy biolmai; \
	fi

# Run all code quality checks
style: format lint

# Run all checks (style + type checking)
check: style mypy

# Run unit tests
test-unit:
	@echo "🧪 Running unit tests..."
	@if [ -d .venv ]; then \
		.venv/bin/python tests/run_tests.py; \
	else \
		python tests/run_tests.py; \
	fi

# Run pipeline tests specifically
test-pipeline:
	@echo "🔬 Running pipeline tests..."
	@if [ -d .venv ]; then \
		.venv/bin/python -m unittest discover tests -p "test_*.py" -v; \
	else \
		python -m unittest discover tests -p "test_*.py" -v; \
	fi

# Run all tests (unit only, no API)
test: test-unit

# Run integration tests (requires API key)
test-integration:
	@echo "🔬 Running integration tests (requires BIOLMAI_TOKEN or BIOLM_API_KEY)..."
	@if [ -z "$$BIOLMAI_TOKEN" ] && [ -z "$$BIOLM_API_KEY" ]; then \
		echo "❌ Error: BIOLMAI_TOKEN or BIOLM_API_KEY environment variable not set"; \
		echo "   Set it with: export BIOLMAI_TOKEN='your-token-here'"; \
		exit 1; \
	fi
	@if [ -d .venv ]; then \
		.venv/bin/python -m unittest tests.test_integration -v; \
	else \
		python -m unittest tests.test_integration -v; \
	fi

# Run all tests (unit + integration)
test-all: test test-integration

# Run tests with pytest (if available)
test-pytest:
	@echo "🧪 Running tests with pytest..."
	@if [ -d .venv ]; then \
		.venv/bin/pytest tests/ -v; \
	else \
		pytest tests/ -v; \
	fi

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"

# Clean everything including venv
clean-all: clean
	@echo "🧹 Removing virtual environment..."
	rm -rf .venv/
	@echo "✅ Everything cleaned!"

# Run examples
example-simple:
	@echo "🚀 Running simple pipeline example..."
	@if [ -d .venv ]; then \
		.venv/bin/python examples/simple_pipeline_example.py; \
	else \
		python examples/simple_pipeline_example.py; \
	fi

example-advanced:
	@echo "🚀 Running advanced pipeline example..."
	@if [ -d .venv ]; then \
		.venv/bin/python examples/advanced_pipeline_example.py; \
	else \
		python examples/advanced_pipeline_example.py; \
	fi

# Show help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
	@echo ""
	@echo "📦 Modern Setup (uv):"
	@echo "  make install          - Install with uv (fast)"
	@echo "  make install-pipeline - Install with pipeline extras"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make install-all      - Install everything"

clean: clean-build clean-pyc clean-test clean-ruff ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -rf {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-ruff: ## remove ruff artifacts
	rm -fr .ruff_cache/

lint/ruff: ## run ruff to check Python code style
	ruff biolmai/ tests/

lint/black: ## run black to check and format Python code style
	black --check biolmai/ tests/

lint/flake8: ## check style with flake8
	flake8 biolmai/ tests/

lint: lint/ruff lint/black lint/flake8 ## check style with ruff, black, flake8

setup-tox:
	pyenv install -s 3.13.3
	pyenv install -s 3.12.10
	pyenv install -s 3.11.6
	pyenv install -s 3.10.13
	pyenv install -s 3.9.22
	pyenv install -s 3.8.20
	pyenv install -s 3.7.17
	pyenv local 3.11.6 3.7.17 3.8.20 3.9.22 3.10.13 3.12.10 3.13.3


test: ## run tests quickly with the default Python
	pytest -s --durations=5 --randomly-seed="$(RS)" $(if $(k),-k "$(k)") -n 0

ptest: ## run tests quickly with the default Python
	pytest -s --durations=5 --randomly-seed="$(RS)" $(if $(k),-k "$(k)") -n $(x)

test-all: ## run tests on every Python version with tox
	tox

test-parallel: ## run tests on every Python version with tox
	tox --parallel 8

coverage: ## check code coverage quickly with the default Python
	coverage run --source biolmai -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	mkdir -p docs/_static docs/api-reference
	rm -f docs/api-reference/modules.rst docs/api-reference/biolmai.rst docs/api-reference/biolmai.*.rst
	sphinx-apidoc -o docs/api-reference biolmai
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

docs-iframe: ## generate docs in iframe mode (no header, for embedding)
	mkdir -p docs/_static docs/api-reference
	rm -f docs/api-reference/modules.rst docs/api-reference/biolmai.rst docs/api-reference/biolmai.*.rst
	sphinx-apidoc -o docs/api-reference biolmai
	$(MAKE) -C docs clean
	IFRAME_MODE=1 $(MAKE) -C docs html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

testrelease: dist ## package and upload a release
	twine upload --repository testpypi --verbose dist/*

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	pip install -e .
	@echo "BioLM Pipeline - Makefile Commands"
	@echo ""
	@echo "📦 Installation:"
	@echo "  make install          - Install core dependencies with uv"
	@echo "  make install-pipeline - Install with pipeline extras"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make install-all      - Install everything (recommended for dev)"
	@echo "  make update           - Update all dependencies"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  make format           - Format code with black"
	@echo "  make lint             - Lint code with ruff"
	@echo "  make mypy             - Type check with mypy"
	@echo "  make style            - Run format + lint"
	@echo "  make check            - Run all checks (style + mypy)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test             - Run all unit tests (no API)"
	@echo "  make test-unit        - Run unit tests"
	@echo "  make test-pipeline    - Run pipeline tests specifically"
	@echo "  make test-pytest      - Run tests with pytest"
	@echo "  make test-integration - Run integration tests (requires API key)"
	@echo "  make test-all         - Run unit + integration tests"
	@echo ""
	@echo "🚀 Examples:"
	@echo "  make example-simple   - Run simple pipeline example"
	@echo "  make example-advanced - Run advanced pipeline example"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean            - Remove build artifacts"
	@echo "  make clean-all        - Remove everything including .venv"
	@echo ""
	@echo "💡 Quick Start:"
	@echo "  1. make install-all   - Install everything"
	@echo "  2. direnv allow       - Activate environment (if using direnv)"
	@echo "  3. make test          - Run tests"
	@echo "  4. make example-simple - Try an example"
