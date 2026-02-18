===================================
Protocol YAML reference
===================================

A **Protocol** defines a workflow for the BioLM server: inputs, tasks (model or gather), and optional MLflow outputs. The schema is validated by the Python client (e.g. ``biolmai protocol validate``).

Top-level keys
--------------

- **name** (required): Protocol name identifier.
- **inputs** (required): Object mapping input names to **InputSpec** objects (see below).
- **tasks** (required): Array of tasks (model tasks or gather tasks).
- **description**: Human-readable description.
- **example_inputs**: Optional object of example values (literals) per input name.
- **progress**: Optional ``{ total_expected: integer | expression }`` for progress tracking.
- **ranking**: Optional ``{ field, order: "ascending"|"descending", top_n }`` for top-N ranking.
- **writing**: Optional ``{ deduplicate?, max_dedupe_size? }`` for output writing.
- **concurrency**: Optional ``{ workflow: integer, tasks: integer }`` for concurrency control.
- **outputs**: Optional array of MLflow output rules (OutputRule).
- **schema_version**: Optional integer (default 1).

Inputs (InputSpec)
------------------

Each value under ``inputs`` is an **InputSpec** object:

- **type**: string, e.g. ``text``, ``float``, ``integer``, ``boolean``, ``select``, ``list_of_str``, ``pdb_text``, ``multiselect``.
- **label**: Optional display label.
- **required** / **optional**: Boolean.
- **help_text**: Optional help string.
- **initial**: Optional default/initial value (literal or expression).
- **min**, **max**: Optional numeric bounds.
- **min_length**, **max_length**: Optional length bounds for text/list.
- **choices**: Optional array of strings (for select/multiselect).
- **advanced**: Optional boolean.
- **step**: Optional number (e.g. for float sliders).

Task forms
----------

**Model task (API task)** — call a model. Use either:

- **Legacy:** ``slug`` + ``action`` (e.g. ``slug: esmfold``, ``action: predict``).
- **Current:** ``app`` + ``class`` + ``method`` (all strings).

Required: **request_body** with at least ``items`` (array, object, or expression) and optional ``params``. Optional: ``response_mapping``, ``depends_on``, ``foreach``, ``skip_if``, ``skip_if_empty``, ``subtasks`` (``count``, ``split_params``).

**Gather task** — collect fields from another task or from an input:

- **type**: ``"gather"``.
- **from**: Task ID or **input name** (key in ``inputs``).
- **fields**: Array of field names to collect.
- **into**: Optional integer.
- **depends_on**, **skip_if_empty**: Optional.

Response mapping
----------------

**response_mapping** values can be:

- A **string expression** (e.g. ``"${{ response.results[*].pdb }}"``), or
- An **object** with:
  - **path**: expression or string.
  - **explode**: optional boolean.
  - **prefix**: optional string.

See ``schema/protocol_schema.json`` for the full JSON Schema and MLflow output rules.
