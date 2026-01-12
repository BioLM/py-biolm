#!/bin/bash
# Demo script to test progress bars with proteinMPNN
# This shell script ensures proper environment setup for tqdm to display correctly

# Set unbuffered output for Python (critical for tqdm)
export PYTHONUNBUFFERED=1

# Use the specific Python environment
PYTHON_BIN="${HOME}/.pyenv/versions/py-biolm/bin/python"

# Check if Python binary exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python binary not found at $PYTHON_BIN"
    exit 1
fi

# Run the Python script
"$PYTHON_BIN" -u ./test_progress_demo.py

