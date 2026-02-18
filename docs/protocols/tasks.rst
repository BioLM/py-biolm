Task forms
==========

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

See :doc:`output` for response mapping and :doc:`about` for top-level structure.
