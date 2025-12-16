``tasks``
=========

The ``tasks`` field defines the workflow steps in a Protocol. Tasks are executed based on their dependencies, forming a directed acyclic graph (DAG) of execution.

Overview
--------

Tasks are the building blocks of protocols. Each task represents a unit of work that:

- Can depend on other tasks
- Can produce outputs used by downstream tasks
- Can be conditional (skip based on conditions)
- Can iterate over arrays (foreach)

There are two types of tasks:

- **Gather tasks**: Collect and batch data from previous tasks
- **Model tasks (API tasks)**: Execute API calls to BioLM models

Schema Definition
-----------------

.. jsonschema:: ../../schema/protocol_schema.json#/properties/tasks

Task Execution Model
--------------------

Tasks execute in **dependency order**, not necessarily the order they appear in the YAML:

1. Tasks with no dependencies run first
2. Tasks wait for their dependencies to complete
3. Tasks can run in parallel if they have no dependency relationship
4. The protocol completes when all tasks finish

**Example dependency graph:**

.. code-block:: yaml

   tasks:
     - id: task_a          # Runs first (no dependencies)
     - id: task_b          # Runs first (no dependencies, parallel with task_a)
     - id: task_c          # Runs after task_a completes
     - id: task_d          # Runs after both task_b and task_c complete

Gather Tasks
------------

Gather tasks collect data from a previous task and batch it for downstream processing. They're essential for:

- **Batching**: Combining individual results into batches for API calls
- **Data transformation**: Selecting and combining specific fields
- **Workflow control**: Creating checkpoints in the workflow

.. jsonschema:: ../../schema/protocol_schema.json#/$defs/GatherTask

**Required fields**:
- ``id``: Unique task identifier
- ``type``: Must be ``"gather"``
- ``from``: Source task ID to gather from
- ``fields``: Array of field names to collect

**Optional fields**:
- ``depends_on``: Explicit dependencies (usually includes ``from`` task)
- ``into``: Batch size (integer or expression)
- ``skip_if_empty``: Skip if source has no results

**How it works**:
1. Collects all results from the ``from`` task
2. Extracts the specified ``fields`` from each result
3. Combines fields into row dictionaries
4. Batches rows into groups of size ``into``
5. Outputs batched results for downstream tasks

**Example:**

.. code-block:: yaml

   - id: sequences_batches_scoring
     type: gather
     from: antifold_generate          # Source task
     fields: [heavy, light]             # Fields to extract
     depends_on: [antifold_generate]   # Wait for source
     into: ${{ max_score_batch }}      # Batch size: 4
     skip_if_empty: true               # Skip if no results

**Common use cases**:
- Batching sequences for scoring APIs
- Combining parallel arrays into row dictionaries
- Creating checkpoints between workflow stages

Model Tasks (API Tasks)
-----------------------

Model tasks execute API calls to BioLM models. They're the primary way to interact with BioLM's model APIs.

**Model Identification**

Model tasks support two identification patterns. Choose based on your needs:

**Pattern 1: ``slug``/``action``** (Recommended for most cases)
  - ``slug``: Model slug (e.g., ``"antifold"``, ``"igbert-paired"``)
  - ``action``: Action name (``predict``, ``encode``, ``generate``, ``similarity``)
  - **Advantage**: Simpler, more readable
  - **Special feature**: ``slug`` can be an expression for dynamic model selection

**Pattern 2: ``class``/``app``/``method``** (Legacy support)
  - ``class``: Model class name (e.g., ``"AntiFoldModel"``)
  - ``app``: Application identifier (e.g., ``"antifold"``)
  - ``method``: Method name (e.g., ``"generate"``, ``"predict"``, ``"predict_log_prob"``)
  - **Use when**: Working with older protocols or specific method requirements

.. jsonschema:: ../../schema/protocol_schema.json#/$defs/ApiTask

**Required fields**:
- ``id``: Unique task identifier
- ``request_body``: API request payload
- ``response_mapping``: Maps API response to protocol fields
- One of: ``slug``/``action`` OR ``class``/``app``/``method``

**Optional fields**:
- ``depends_on``: Task dependencies
- ``foreach``: Iterate over array, creating subtasks
- ``skip_if``: Conditional execution expression
- ``skip_if_empty``: Skip if dependencies are empty
- ``fail_on_error``: Whether to fail on errors (default: ``true``)

Request Body
------------

The ``request_body`` defines what to send to the API. It has two main parts:

.. jsonschema:: ../../schema/protocol_schema.json#/$defs/RequestBody

**items** (array or expression)
  The input items to process. Can be:
  - A literal array: ``[{pdb: "...", chain: "A"}]``
  - A template expression: ``${{ previous_task.results }}``
  - An item reference in foreach: ``${{ item }}``

**params** (object, optional)
  Additional parameters for the API call. Common examples:
  - ``temperature``: Sampling temperature
  - ``num_seq_per_target``: Number of sequences to generate
  - ``top_k``: Top-k sampling parameter

**Example:**

.. code-block:: yaml

   request_body:
     items:
       - pdb: ${{ pdb_str }}
         chain: ${{ heavy_chain }}
     params:
       temperature: ${{ temperature }}
       num_seq_per_target: ${{ n_samples }}

Response Mapping
----------------

The ``response_mapping`` extracts fields from the API response and makes them available to downstream tasks.

.. jsonschema:: ../../schema/protocol_schema.json#/$defs/ResponseMapping

**How it works**:
- Each key becomes a field name available to downstream tasks
- Each value is a JSONPath expression that extracts data from the response
- JSONPath expressions use ``${{ response.path.to.field }}`` syntax
- Array wildcards ``[*]`` extract from all array elements

**JSONPath syntax**:
- ``${{ response.results[0].field }}`` - First result's field
- ``${{ response.results[*].field }}`` - Field from all results
- ``${{ response.results[*].sequences[*].heavy }}`` - Nested array extraction

**Example:**

.. code-block:: yaml

   response_mapping:
     heavy: "${{ response.results[*].sequences[*].heavy }}"
     light: "${{ response.results[*].sequences[*].light }}"
     score: "${{ response.results[*].score }}"

**Important**: Field names in ``response_mapping`` become available to:
- Downstream tasks (via ``depends_on``)
- Output rules (in the ``outputs`` field)
- Other tasks via ``${{ task_id.field_name }}``

Examples
--------

Basic Model Task
~~~~~~~~~~~~~~~~

Simple task with slug/action:

.. code-block:: yaml

   - id: igbert_score
     slug: igbert-paired
     action: predict
     request_body:
       items: ${{ sequences }}
     response_mapping:
       log_prob: "${{ response.results[*].log_prob }}"

Model Task with Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Task that waits for another task:

.. code-block:: yaml

   - id: igbert_score
     slug: igbert-paired
     action: predict
     depends_on: [sequences_batches_scoring]
     request_body:
       items: ${{ sequences_batches_scoring.results }}
     response_mapping:
       log_prob: "${{ response.results[*].log_prob }}"

Model Task with Foreach (Fan-out)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create multiple subtasks, one per batch:

.. code-block:: yaml

   - id: igbert_score
     slug: igbert-paired
     action: predict
     depends_on: [sequences_batches_scoring]
     foreach: ${{ sequences_batches_scoring.results }}  # One task per batch
     request_body:
       items: ${{ item }}  # Use the batch as input
     response_mapping:
       log_prob: "${{ response.results[*].log_prob }}"

Model Task with Conditional Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip task if condition is met:

.. code-block:: yaml

   - id: igbert_score
     slug: igbert-paired
     action: predict
     depends_on: [sequences_batches_scoring]
     skip_if: "${{ sequences_batches_scoring.empty }}"  # Skip if no data
     foreach: ${{ sequences_batches_scoring.results }}
     request_body:
       items: ${{ item }}
     response_mapping:
       log_prob: "${{ response.results[*].log_prob }}"

Model Task with Dynamic Slug (Multi-model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use expression in slug to select model dynamically:

.. code-block:: yaml

   - id: multi_model_score
     slug: ${{ selected_model }}  # Can be "igbert-paired", "esm2", etc.
     action: predict
     request_body:
       items: ${{ sequences }}
     response_mapping:
       score: "${{ response.results[*].score }}"

Model Task with Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continue execution even if this task fails:

.. code-block:: yaml

   - id: optional_folding
     slug: abodybuilder3
     action: predict
     fail_on_error: false  # Don't fail protocol if this fails
     request_body:
       items: ${{ sequences }}
     response_mapping:
       pdb: "${{ response.results[*].pdb }}"

Legacy Model Task (class/app/method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the older identification pattern:

.. code-block:: yaml

   - id: antifold_generate
     class: AntiFoldModel
     app: antifold
     method: generate
     request_body:
       items:
         - pdb: ${{ pdb_str }}
       params:
         num_seq_per_target: ${{ n_samples }}
     response_mapping:
       heavy: "${{ response.results[*].sequences[*].heavy }}"
       light: "${{ response.results[*].sequences[*].light }}"

Task Dependencies
----------------

Tasks can specify dependencies in several ways:

**depends_on** (array of task IDs)
  Explicitly list tasks that must complete before this task runs. The task waits for ALL listed tasks to finish.

**foreach** (expression)
  Iterate over an array, creating one subtask per array element. The subtask has access to ``${{ item }}`` for the current element.

**skip_if** (expression)
  Conditionally skip the task. If the expression evaluates to ``true``, the task is skipped entirely.

**skip_if_empty** (boolean)
  Skip the task if any dependency has no results. Useful for gather tasks and downstream processing.

**Best practices**:
- Always include ``depends_on`` for clarity, even if implicit
- Use ``skip_if_empty`` for optional processing steps
- Use ``skip_if`` for complex conditional logic
- Use ``foreach`` to parallelize batch processing

Common Patterns
---------------

**Generate → Batch → Score pattern**:

.. code-block:: yaml

   - id: generate
     slug: antifold
     action: generate
     request_body:
       items: [{pdb: ${{ pdb_str }}}]
     response_mapping:
       sequences: "${{ response.results[*].sequences }}"
   
   - id: batch_sequences
     type: gather
     from: generate
     fields: [sequences]
     into: 4
   
   - id: score
     slug: igbert-paired
     action: predict
     depends_on: [batch_sequences]
     foreach: ${{ batch_sequences.results }}
     request_body:
       items: ${{ item }}
     response_mapping:
       score: "${{ response.results[*].score }}"

**Parallel processing pattern**:

.. code-block:: yaml

   - id: task_a
     # ... runs in parallel with task_b
   
   - id: task_b
     # ... runs in parallel with task_a
   
   - id: task_c
     depends_on: [task_a, task_b]  # Waits for both
