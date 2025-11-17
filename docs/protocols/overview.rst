.. _protocols-overview:

Overview
========

BioLM Protocols allow you to define complex workflows and configurations using structured YAML files.

Documentation Structure
-----------------------

**Where to write Protocol documentation:**

- **Usage guides:** ``docs/protocols/usage/*.rst`` - Write usage guides for each topic (inputs, tasks, outputs, settings)
- **This overview page:** ``docs/protocols/overview.rst`` - General Protocol information and overview

**Auto-generated documentation (planned):**

- **YAML schema reference:** ``docs/protocols/yaml-reference/index.rst`` - Will be auto-generated from JSON Schema (currently manually written)

**How to update Protocol documentation:**

1. **YAML schema reference (planned for auto-generation):** Once JSON Schema is implemented, this will be auto-generated from the schema file. Currently, edit ``docs/protocols/yaml-reference/index.rst`` manually.

2. **Usage guides (manually written):** Edit the files in ``docs/protocols/usage/`` to add examples, explanations, and tutorials.

3. **Overview (manually written):** Edit this file (``docs/protocols/overview.rst``) for general Protocol information.

For more details, see the :doc:`../authoring-guide` section.

