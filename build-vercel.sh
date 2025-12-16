#!/bin/bash
set -e

# Set locale to avoid locale errors in containerized environments
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Install dependencies with Python 3.12 compatible versions
pip install --upgrade pip setuptools wheel
pip install -r requirements_vercel.txt

# Verify critical packages are installed (check both import names)
python -c "import sphinx_jsonschema" 2>/dev/null || python -c "import sphinx_jsonschema as sj" 2>/dev/null || {
    echo "WARNING: sphinx-jsonschema not found, attempting to reinstall..."
    pip install --force-reinstall --no-cache-dir sphinx-jsonschema>=1.17.0
    python -c "import sphinx_jsonschema" || {
        echo "ERROR: sphinx-jsonschema still not available after reinstall"
        pip list | grep -i jsonschema || echo "No jsonschema packages found"
        exit 1
    }
}

# Install the package itself so sphinx-apidoc can import modules
pip install -e .

# Build HTML docs in iframe mode using Makefile target (same as GitHub Actions)
make docs-iframe

# Verify build output exists
if [ ! -d "docs/_build/html" ]; then
    echo "ERROR: Build output directory docs/_build/html does not exist"
    exit 1
fi

if [ ! -f "docs/_build/html/index.html" ]; then
    echo "ERROR: Build output index.html does not exist"
    exit 1
fi

echo "Build completed successfully. Output in docs/_build/html"

