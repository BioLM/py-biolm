``tasks``
=========

The ``tasks`` field defines the workflow steps in a Protocol. Tasks are executed in order, with dependencies controlling execution flow.

Overview
--------

Tasks can be one of two types:
- **Gather tasks**: Collect and batch data from previous tasks
- **Model tasks (API tasks)**: Execute API calls to BioLM models

Schema Definition
-----------------

.. jsonschema:: ../../../schema/protocol_schema.json#/properties/tasks

Gather Tasks
------------

Gather tasks collect and batch data from previous tasks.

.. jsonschema:: ../../../schema/protocol_schema.json#/$defs/GatherTask

**Example:**

.. code-block:: yaml

   - id: sequences_batches_scoring
     type: gather
     from: antifold_generate
     fields: [heavy, light]
     depends_on: [antifold_generate]
     into: ${{ max_score_batch }}
     skip_if_empty: true

Model Tasks (API Tasks)
-----------------------

Model tasks execute API calls to BioLM models. They support two identification patterns:

**Pattern 1: ``slug``/``action``**
  - ``slug``: Model slug (e.g., ``"antifold"``, ``"igbert-paired"``) - can be an expression for fan-out
  - ``action``: Action name (``predict``, ``encode``, ``generate``, ``similarity``)

**Pattern 2: ``class``/``app``/``method``**
  - ``class``: Model class name (e.g., ``"AntiFoldModel"``)
  - ``app``: Application identifier (e.g., ``"antifold"``)
  - ``method``: Method name (e.g., ``"generate"``, ``"predict"``, ``"predict_log_prob"``)

.. jsonschema:: ../../../schema/protocol_schema.json#/$defs/ApiTask

Request Body
~~~~~~~~~~~~

The ``request_body`` field defines the API request payload:

.. jsonschema:: ../../../schema/protocol_schema.json#/$defs/RequestBody

Response Mapping
~~~~~~~~~~~~~~~~~

The ``response_mapping`` field maps API response fields to protocol fields:

.. jsonschema:: ../../../schema/protocol_schema.json#/$defs/ResponseMapping

Examples
--------

Model Task with slug/action
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   - id: igbert_score
     slug: igbert-paired
     action: predict
     depends_on: [sequences_batches_scoring]
     foreach: ${{ sequences_batches_scoring.results }}
     skip_if: "${{ sequences_batches_scoring.empty }}"
     request_body:
       items: ${{ item }}
     response_mapping:
       log_prob: "${{ response.results[*].log_prob }}"

Model Task with class/app/method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   - id: antifold_generate
     class: AntiFoldModel
     app: antifold
     method: generate
     request_body:
       items:
         - pdb: ${{ pdb_str }}
       params:
         num_seq_per_target: ${{ n_samples // execution.concurrency.workflow }}
     response_mapping:
       heavy: "${{ response.results[*].sequences[*].heavy }}"
       light: "${{ response.results[*].sequences[*].light }}"
     fail_on_error: true

Model Task with Slug Expression (Fan-out)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   - id: multi_model_score
     slug: ${{ selected_model }}  # Expression allows multiselect/fan-out
     action: predict
     request_body:
       items: ${{ sequences }}
     response_mapping:
       score: "${{ response.results[*].score }}"

Task Dependencies
----------------

Tasks can specify dependencies using:

- **depends_on**: Array of task IDs that must complete before this task runs
- **foreach**: Iterate over array results, creating subtasks
- **skip_if**: Conditional execution expression
