#!/usr/bin/env python

"""The setup script."""

import os
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=6.0",
    "requests",
    "httpx>=0.23.0",
    "httpcore",
    "synchronicity>=0.5.0; python_version >= '3.9'",
    "synchronicity<0.5.0; python_version < '3.9'",
    "typing_extensions; python_version < '3.9'",
    "aiodns",
    "aiohttp<=3.8.6; python_version < '3.12'",
    "aiohttp>=3.9.0; python_version >= '3.12'",
    "async-lru",
    "aiofiles",
    "cryptography",
    "rich>=13.0.0",
    "jsonschema>=4.0.0",
    "pyyaml>=5.0",
]

test_requirements = [
    "pytest>=3",
]


def _suppress_urllib3_warnings(self):
    """Modify generated entry point script to suppress urllib3 warnings."""
    import sys
    # Find the scripts directory - try multiple locations
    possible_dirs = []
    if hasattr(self, 'install_scripts') and self.install_scripts:
        possible_dirs.append(self.install_scripts)
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a virtualenv
        possible_dirs.append(os.path.join(sys.prefix, 'bin'))
    possible_dirs.append(os.path.join(sys.prefix, 'bin'))
    # Also check common venv locations
    if 'venv' in sys.prefix or 'virtualenv' in sys.prefix:
        possible_dirs.append(os.path.join(sys.prefix, 'bin'))
    
    for scripts_dir in possible_dirs:
        script_path = os.path.join(scripts_dir, 'biolm')
        if os.path.exists(script_path):
            with open(script_path, 'r') as f:
                content = f.read()
            # Add stderr redirection to suppress urllib3 warnings
            if 'StringIO' not in content:
                new_content = content.replace(
                    'from biolmai.cli_entry import cli',
                    'import sys\nfrom io import StringIO\n_original_stderr = sys.stderr\nsys.stderr = StringIO()\nfrom biolmai.cli_entry import cli\nsys.stderr = _original_stderr'
                )
                with open(script_path, 'w') as f:
                    f.write(new_content)
                break

class PostInstallCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        _suppress_urllib3_warnings(self)

class PostInstallCommandInstall(install):
    """Post-installation for install mode."""
    def run(self):
        install.run(self)
        _suppress_urllib3_warnings(self)

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
            "biolm=biolmai.cli_entry:cli",
        ],
        'mlflow.request_header_provider': [
            'unused=biolmai.core.seqflow_auth:BiolmaiRequestHeaderProvider',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords=["biolmai", "biolm", "bioai", "bio-ai", "bio-lm", "bio-llm", "bio-language-model", "bio-language-models-api", "python-client"],
    name="biolmai",
    packages=find_packages(include=["biolmai", "biolmai.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/BioLM/py-biolm",
    version='0.2.8',
    zip_safe=False,
    cmdclass={
        'develop': PostInstallCommand,
        'install': PostInstallCommandInstall,
    },
)
