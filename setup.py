#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

setup(
    author="BioLM",
    author_email="support@biolm.ai",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    description="BioLM Python client",
    entry_points={
        "console_scripts": [
            "biolmai=biolmai.cli_entry:cli",
        ],
        'mlflow.request_header_provider': [
            'unused=biolmai.core.seqflow_auth:BiolmaiRequestHeaderProvider',
        ],
    },
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords=["biolmai", "biolm", "bioai", "bio-ai", "bio-lm", "bio-llm", "bio-language-model", "bio-language-models-api", "python-client"],
    name="biolmai",
    packages=find_packages(include=["biolmai", "biolmai.*"]),
    test_suite="tests",
    url="https://github.com/BioLM/py-biolm",
    version='0.4.0',
    zip_safe=False,
)
