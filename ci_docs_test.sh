#!/bin/bash
set -ev
if [ "${TRAVIS_PYTHON_VERSION}" = "3.9" ]; then
  sudo apt-get update && sudo apt-get install -y font-manager || true
  tox -e build_docs
fi
