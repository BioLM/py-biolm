Inputs (InputSpec)
==================

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

See :doc:`about` for top-level structure and :doc:`schema` for the full JSON Schema.
