#!/bin/bash
set -e

# Install dependencies with Python 3.12 compatible versions
pip install --upgrade pip setuptools wheel
pip install -r requirements_vercel.txt

# Generate API docs
mkdir -p docs/_static docs/api-reference
rm -f docs/biolmai.rst docs/api-reference/biolmai.rst docs/api-reference/modules.rst
sphinx-apidoc -o docs/api-reference biolmai

# Build HTML docs
cd docs
make clean
make html

