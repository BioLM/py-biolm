#!/bin/bash
set -ev
if [ "${TRAVIS_PYTHON_VERSION}" = "3.9" ]; then
  pip install -r requirements_docs.txt
  make docs
fi
