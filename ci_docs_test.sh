#!/bin/bash
set -ev
if [ "${TRAVIS_PYTHON_VERSION}" = "3.9.0" ]; then
  make docs
fi
