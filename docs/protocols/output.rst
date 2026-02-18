Response mapping and outputs
=============================

On model tasks, *response_mapping* can be a string expression (e.g. to pull a field from the API response) or an object with *path*, optional *explode*, and optional *prefix*. The top-level *outputs* array defines MLflow output rules for logging. See :doc:`schema` for the full JSON Schema and MLflow rules.

**Example (response_mapping):**

.. code-block:: yaml

    - id: predict
      slug: esmfold
      action: predict
      request_body:
        items: "{{ inputs.sequences }}"
      response_mapping: "{{ response.results[*].pdb }}"
