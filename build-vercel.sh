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

# Build HTML docs in iframe mode using Makefile target (same as GitHub Actions)
make docs-iframe

