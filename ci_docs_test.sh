#!/bin/bash
set -ev
if [ "${TRAVIS_PYTHON_VERSION}" = "3.11-dev" ]; then
  sudo apt-get update && sudo apt-get install -y font-manager fonts-roboto fonts-noto || true
  tox -e build_docs
fi
