#!/bin/bash
if [ "${TRAVIS_PYTHON_VERSION}" = "3.9.9" ]; then
  make docs
fi