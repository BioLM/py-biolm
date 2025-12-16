.. _protocols-overview:

Overview
========

BioLM Protocols allow you to define complex workflows and configurations using structured YAML files.

Documentation Structure
-----------------------

**Where to write Protocol documentation:**

- **Usage guides:** ``docs/protocols/usage/*.rst`` - Write usage guides for each topic (inputs, tasks, outputs, settings)
- **This overview page:** ``docs/protocols/overview.rst`` - General Protocol information and overview

**Auto-generated documentation:**

- **Protocol schema pages:** ``docs/protocols/*.rst`` - Auto-generated from JSON Schema (about, inputs, execution, tasks, output)

**How to update Protocol documentation:**

1. **Protocol schema pages (auto-generated):** Schema documentation is auto-generated from ``schema/protocol_schema.json``. To update, modify the schema file and rebuild docs.

2. **Usage guides (manually written):** Edit the files in ``docs/protocols/usage/`` to add examples, explanations, and tutorials.

3. **Overview (manually written):** Edit this file (``docs/protocols/overview.rst``) for general Protocol information.

For more details, see the :doc:`../authoring-guide` section.

