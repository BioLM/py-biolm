#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=6.0",
    "requests",
    "httpx>=0.23.0",
    "httpcore",
    "synchronicity>=0.5.0",
    "aiodns",
    "aiohttp<=3.8.6; python_version < '3.12'",
    "aiohttp>=3.9.0; python_version >= '3.12'",
    "async-lru",
    "aiofiles",
]

test_requirements = [
    "pytest>=3",
]

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
            "biolmai=biolmai.cli:cli",
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="biolmai",
    name="biolmai",
    packages=find_packages(include=["biolmai", "biolmai.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/BioLM/py-biolm",
    version='0.2.4',
    zip_safe=False,
)
