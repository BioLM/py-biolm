#!/bin/bash
pip install -e ".[dev]"
make dist
twine check dist/*
make release
