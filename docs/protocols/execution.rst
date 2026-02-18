Execution
=========

Tasks run in order. You can make a task wait for others (depends_on), run conditionally (skip_if, skip_if_empty), or repeat over a list (foreach). See :doc:`tasks` for task definitions and :doc:`schema` for the full JSON Schema.

**Example (depends_on):**

.. code-block:: yaml

    tasks:
      - id: encode
        slug: esm2-8m
        action: encode
        request_body:
          items: "{{ inputs.sequences }}"
      - id: predict
        slug: esmfold
        action: predict
        depends_on: [encode]
        request_body:
          items: "{{ tasks.encode.response.results }}"
