#!/bin/bash
set -ev
if [ "${TRAVIS_PYTHON_VERSION}" = "3.9" ]; then
  sudo apt-get update && sudo apt-get install -y font-manager msttcorefont
  pip install -r requirements_docs.txt
  make docs
fi
