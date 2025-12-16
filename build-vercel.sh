#!/bin/bash
set -e

# Set locale to avoid locale errors in containerized environments
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Install dependencies with Python 3.12 compatible versions
pip install --upgrade pip setuptools wheel
pip install -r requirements_vercel.txt

# Install the package itself so sphinx-apidoc can import modules
pip install -e .

# Generate API docs
mkdir -p docs/_static docs/api-reference
rm -f docs/biolmai.rst docs/api-reference/biolmai.rst docs/api-reference/modules.rst
# Try python module form first, fallback to direct command if available
python -m sphinx.ext.apidoc -o docs/api-reference biolmai 2>/dev/null || \
python -m sphinx.apidoc -o docs/api-reference biolmai 2>/dev/null || \
sphinx-apidoc -o docs/api-reference biolmai

# Build HTML docs in iframe mode
cd docs
make clean
IFRAME_MODE=1 make html

