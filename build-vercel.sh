#!/bin/bash
set -e

# Set locale to avoid locale errors in containerized environments
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Install dependencies with Python 3.12 compatible versions
pip install --upgrade pip setuptools wheel
pip install -r requirements_vercel.txt

# Verify sphinx-jsonschema is installed (package may be installed but import check can fail due to path issues)
# The package is installed via requirements_vercel.txt, so we just verify it's in the package list
if ! pip show sphinx-jsonschema >/dev/null 2>&1; then
    echo "WARNING: sphinx-jsonschema not found in pip, attempting to install..."
    pip install sphinx-jsonschema>=1.17.0
fi

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

