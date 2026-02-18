Response mapping and outputs
=============================

**response_mapping** (on model tasks) values can be:

- A **string expression** (e.g. ``"${{ response.results[*].pdb }}"``), or
- An **object** with:
  - **path**: expression or string.
  - **explode**: optional boolean.
  - **prefix**: optional string.

**outputs**: Optional array of MLflow output rules (OutputRule) for logging results to MLflow.

See :doc:`schema` for the full JSON Schema and MLflow output rules.
