Full JSON Schema
================

The formal JSON Schema for BioLM Protocol YAML (inputs, tasks, execution, MLflow outputs) is shipped with the Python client and used for validation (e.g. ``biolmai protocol validate``). It covers the structure described in this section: :doc:`about`, :doc:`inputs`, :doc:`tasks`, :doc:`execution`, and :doc:`output`.

The schema file (``protocol_schema.json``) is available in the package under the ``schema`` directory in the source tree. User-facing protocol task keys are ``slug`` and ``action``; see :doc:`tasks`.
