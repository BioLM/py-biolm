#!/bin/bash
set -e

# Run from repo root (Vercel may run from different cwd)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Set locale to avoid locale errors in containerized environments
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Vercel uses uv-managed Python (PEP 668); use a venv so we don't modify system packages
python3 -m venv .venv
source .venv/bin/activate
PYTHON="${PYTHON:-python}"
PIP="$PYTHON -m pip"

# Upgrade pip with cap to avoid breaking setuptools
$PIP install --upgrade "pip<25" setuptools wheel

# Install docutils first with version constraint to avoid conflicts
$PIP install "docutils>=0.20.0,<0.22"

# Install remaining dependencies
$PIP install -r requirements_vercel.txt

# Verify sphinx-jsonschema is installed
if ! $PIP show sphinx-jsonschema >/dev/null 2>&1; then
    echo "WARNING: sphinx-jsonschema not found, installing..."
    $PIP install "sphinx-jsonschema>=1.17.0"
fi

# Install the package itself so sphinx-apidoc can import modules
$PIP install -e .

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

