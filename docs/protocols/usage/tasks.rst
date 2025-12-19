``tasks``
=========

The ``tasks`` field defines the workflow steps in a Protocol. Tasks are executed in order, with dependencies controlling execution flow.

Overview
--------

Tasks can be one of two types:
- **Gather tasks**: Collect and batch data from previous tasks
- **Model tasks (API tasks)**: Execute API calls to BioLM models

Gather Tasks
------------

Gather tasks collect and batch data from previous tasks.

**Properties**:
- **id** (string, required): Unique task identifier
- **type** (string, required): Must be ``"gather"``
- **from** (string, required): Source task ID to gather from
- **fields** (array of strings, required): Field names to collect
- **into** (integer or expression, optional): Batch size
- **depends_on** (array of strings, optional): Task dependencies
- **skip_if_empty** (boolean, optional): Skip if dependencies are empty

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

**Properties**:
- **id** (string, required): Unique task identifier
- **request_body** (object, required): API request payload
- **One of** (required):
  - **slug** (string or expression) + **action** (string)
  - **class** (string) + **app** (string) + **method** (string)
- **response_mapping** (object, optional): Maps API response to protocol fields
- **depends_on** (array of strings, optional): Task dependencies
- **foreach** (string or expression, optional): Iterate over array
- **skip_if** (string or expression, optional): Conditional execution
- **skip_if_empty** (boolean, optional): Skip if dependencies are empty
- **fail_on_error** (boolean, optional): Whether to fail on errors (default: ``true``)

Request Body
~~~~~~~~~~~~

The ``request_body`` field defines the API request payload:

**Properties**:
- **items** (array or expression, required): Input items to process
- **params** (object, optional): Additional parameters for the API call

Response Mapping
~~~~~~~~~~~~~~~~~

The ``response_mapping`` field maps API response fields to protocol fields:

**Properties**:
- Object with string keys and JSONPath expression values
- Each key becomes a field name available to downstream tasks
- Each value is a JSONPath expression: ``"${{ response.path.to.field }}"``

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
