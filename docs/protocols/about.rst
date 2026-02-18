About Protocols
===============

A **Protocol** defines a workflow for the BioLM server: inputs, tasks (model or gather), and optional MLflow outputs. The schema is validated by the Python client (e.g. ``biolmai protocol validate``).

Top-level keys
--------------

- **name** (required): Protocol name identifier.
- **inputs** (required): Object mapping input names to **InputSpec** objects (see :doc:`inputs`).
- **tasks** (required): Array of tasks (model tasks or gather tasks). See :doc:`tasks` and :doc:`execution`.
- **description**: Human-readable description.
- **example_inputs**: Optional object of example values (literals) per input name.
- **progress**: Optional ``{ total_expected: integer | expression }`` for progress tracking.
- **ranking**: Optional ``{ field, order: "ascending"|"descending", top_n }`` for top-N ranking.
- **writing**: Optional ``{ deduplicate?, max_dedupe_size? }`` for output writing.
- **concurrency**: Optional ``{ workflow: integer, tasks: integer }`` for concurrency control.
- **outputs**: Optional array of MLflow output rules (OutputRule). See :doc:`output`.
- **schema_version**: Optional integer (default 1).

See ``schema/protocol_schema.json`` for the full JSON Schema and MLflow output rules.
