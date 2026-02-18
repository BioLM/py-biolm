Inputs (InputSpec)
==================

Each entry under ``inputs`` is an InputSpec: a ``type`` (e.g. text, float, integer, boolean, select, list_of_str, pdb_text, multiselect) plus optional ``label``, ``required``/``optional``, ``help_text``, ``initial``, ``min``/``max``, ``min_length``/``max_length``, ``choices`` (for select/multiselect), ``advanced``, and ``step``.

**Example:**

.. code-block:: yaml

    inputs:
      sequences:
        type: list_of_str
        label: Protein sequences
        required: true
      temperature:
        type: float
        label: Sampling temperature
        initial: 0.7
        min: 0
        max: 2

See :doc:`about` for top-level structure and :doc:`schema` for the full JSON Schema.
