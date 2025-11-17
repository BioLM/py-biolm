Documentation Authoring Guide
==============================

This guide explains where to write documentation and where auto-generated documentation appears.

Documentation Structure
-----------------------

**Manually Written Documentation:**
  - Getting Started guides (``docs/getting-started/``)
  - CLI usage guides (``docs/cli/usage/``)
  - SDK usage guides (``docs/sdk/usage/``)
  - Protocol usage guides (``docs/protocols/usage/``)
  - Overview pages (``docs/cli/overview.rst``, ``docs/sdk/overview.rst``, ``docs/protocols/overview.rst``)
  - Resources (``docs/resources/``)

**Auto-Generated Documentation:**
  - CLI reference (``docs/cli/reference.rst``) - Generated from ``biolmai.cli`` using ``sphinx-click``
  - SDK API reference (``docs/sdk/api-reference/``) - Generated from Python code using ``sphinx-apidoc``
  - Protocol YAML reference (``docs/protocols/yaml-reference/index.rst``) - Generated from JSON Schema (when implemented)

Where to Write Documentation
----------------------------

CLI Documentation
~~~~~~~~~~~~~~~~~

**Write here:**
  - ``docs/cli/overview.rst`` - CLI overview and general information
  - ``docs/cli/usage/*.rst`` - Usage guides for each topic (authenticating, workspaces, models, protocols, datasets)

**Auto-generated:**
  - ``docs/cli/reference.rst`` - Command reference (uses ``sphinx-click`` directives pointing to ``biolmai.cli``)

**Note:** The CLI reference page uses ``.. click::`` directives that pull documentation from the Click command definitions in ``biolmai/cli.py``. To update command documentation, edit the docstrings and help text in the CLI code.

SDK Documentation
~~~~~~~~~~~~~~~~~

**Write here:**
  - ``docs/sdk/overview.rst`` - SDK overview and general information
  - ``docs/sdk/usage/*.rst`` - Usage guides for each topic (authenticating, workspaces, models, protocols, datasets, batching, error-handling, async-sync)

**Auto-generated:**
  - ``docs/sdk/api-reference/`` - API reference (generated from Python docstrings using ``sphinx-apidoc``)

**Note:** The API reference is generated from docstrings in the Python code. To update API documentation, edit docstrings in ``biolmai/*.py`` files.

Protocol Schema Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Write here:**
  - ``docs/protocols/overview.rst`` - Protocol overview and general information
  - ``docs/protocols/usage/*.rst`` - Usage guides (inputs, tasks, outputs, settings)

**Auto-generated (planned):**
  - ``docs/protocols/yaml-reference/index.rst`` - YAML schema reference (will be generated from JSON Schema)

**Note:** Currently, the YAML reference is manually written. Once the JSON Schema is implemented, this should be auto-generated from the schema file.

Getting Started Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Write here:**
  - ``docs/getting-started/installation.rst`` - Installation instructions
  - ``docs/getting-started/authentication.rst`` - Authentication setup
  - ``docs/getting-started/quickstart.rst`` - Quick start guide
  - ``docs/getting-started/features.rst`` - Feature overview

Resources
~~~~~~~~~

**Write here:**
  - ``docs/resources/rest-api.rst`` - REST API documentation
  - ``docs/resources/guides/*.rst`` - Additional guides (notebooks, tutorials, etc.)

Building Documentation
----------------------

To build the documentation:

.. code-block:: bash

   make docs

This will:
1. Generate API docs using ``sphinx-apidoc``
2. Build HTML documentation using Sphinx
3. Open the documentation in your browser

Auto-Generation Details
-----------------------

CLI Reference (sphinx-click)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CLI reference uses the ``sphinx-click`` extension to auto-generate documentation from Click commands.

**Location:** ``docs/cli/reference.rst``

**Source:** ``biolmai/cli.py``

**How it works:**
  - Uses ``.. click::`` directives to reference Click command groups and commands
  - Pulls docstrings and help text from the Click command definitions
  - To update: Edit docstrings in ``biolmai/cli.py``

**Example:**
  .. code-block:: rst

     .. click:: biolmai.cli:cli
        :prog: biolmai
        :show-nested:

SDK API Reference (sphinx-apidoc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SDK API reference is generated from Python docstrings.

**Location:** ``docs/sdk/api-reference/``

**Source:** ``biolmai/*.py``

**How it works:**
  - ``sphinx-apidoc`` scans the ``biolmai/`` package
  - Generates ``.rst`` files from module docstrings
  - Uses ``sphinx.ext.autodoc`` to render docstrings in HTML
  - To update: Edit docstrings in Python source files

**Build command:**
  .. code-block:: bash

     sphinx-apidoc -o docs/api-reference biolmai

Protocol YAML Reference (planned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Protocol YAML reference will be auto-generated from JSON Schema.

**Location:** ``docs/protocols/yaml-reference/index.rst``

**Source:** JSON Schema file (to be created)

**How it will work:**
  - A script or Sphinx extension will process the JSON Schema
  - Generate documentation for all fields, types, and validation rules
  - To update: Edit the JSON Schema file

**Status:** Currently manually written. Auto-generation to be implemented.

Best Practices
--------------

1. **Keep auto-generated docs separate:** Don't manually edit files in ``docs/sdk/api-reference/`` or ``docs/cli/reference.rst`` (except for the directive setup)

2. **Document in code:** For API and CLI reference, add docstrings and help text directly in the source code

3. **Write usage guides manually:** Usage guides (``docs/*/usage/*.rst``) should be manually written with examples and explanations

4. **Keep overviews concise:** Overview pages should provide high-level information and point to detailed usage guides

5. **Update this guide:** When adding new auto-generation, update this guide to document the process

