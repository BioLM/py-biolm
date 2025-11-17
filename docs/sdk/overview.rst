.. _sdk-overview:

Overview
========

The BioLM Python SDK provides a high-level, user-friendly interface for interacting with the BioLM API.

Documentation Structure
-----------------------

**Where to write SDK documentation:**

- **Usage guides:** ``docs/sdk/usage/*.rst`` - Write usage guides for each topic (authenticating, workspaces, models, protocols, datasets, batching, error-handling, async-sync)
- **This overview page:** ``docs/sdk/overview.rst`` - General SDK information and overview

**Auto-generated documentation:**

- **API reference:** ``docs/sdk/api-reference/`` - Auto-generated from Python docstrings using ``sphinx-apidoc``

**How to update SDK documentation:**

1. **API reference (auto-generated):** Edit docstrings in ``biolmai/*.py`` files. The ``sphinx-apidoc`` tool will generate reference documentation from these docstrings.

2. **Usage guides (manually written):** Edit the files in ``docs/sdk/usage/`` to add examples, explanations, and tutorials.

3. **Overview (manually written):** Edit this file (``docs/sdk/overview.rst``) for general SDK information.

For more details, see the :doc:`../authoring-guide` section.
