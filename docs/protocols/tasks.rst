Task forms
==========

**Model task** — calls a model. Use ``slug`` and ``action`` (e.g. esmfold / predict) or ``app``, ``class``, and ``method``. The request body must include ``items`` (array, object, or expression) and can include ``params``. Optional: ``response_mapping``, ``depends_on``, ``foreach``, ``skip_if``, ``skip_if_empty``, ``subtasks``.

**Example (model task):**

.. code-block:: yaml

    - id: predict
      slug: esmfold
      action: predict
      request_body:
        items: "{{ inputs.sequences }}"
        params: {}

**Gather task** — collects fields from another task or from an input. Set ``type`` to "gather", ``from`` to a task ID or input name, and ``fields`` to the list of field names. Optional: ``into``, ``depends_on``, ``skip_if_empty``.

**Example (gather task):**

.. code-block:: yaml

    - type: gather
      from: predict
      fields: [pdb, mean_plddt]

See :doc:`output` for response mapping and :doc:`about` for top-level structure.
