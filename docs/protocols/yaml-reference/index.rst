YAML Reference
==============

Complete reference documentation for Protocol YAML schema. Protocols use JSON Schema (Draft 2020-12) for validation.

.. note::

   **Documentation authors:** This page is currently manually written. Once JSON Schema is implemented, this should be auto-generated from the schema file. See :doc:`../../authoring-guide` for details.

Schema Overview
---------------

Protocols are validated against a formal JSON Schema that defines:

- **Top-level structure**: Required and optional fields
- **Field types**: Data types and constraints for each field
- **Validation rules**: Conditional requirements, enum values, patterns
- **Task definitions**: Model tasks and gather tasks

Top-Level Schema
----------------

**Required Fields:**
  - ``name`` (string) - Protocol name identifier
  - ``inputs`` (object) - Input parameter definitions with default values
  - ``tasks`` (array) - List of workflow tasks to execute

**Optional Fields:**
  - ``description`` (string) - Human-readable description
  - ``outputs`` (array) - Select which task outputs to log to MLflow
  - ``execution`` (object) - Execution configuration

Field Definitions
-----------------

name
~~~~

- **Type**: ``string``
- **Required**: Yes
- **Description**: Protocol name identifier
- **Constraints**: Non-empty string
- **Example**: ``"Example_Protocol_YAML"``

description
~~~~~~~~~~~

- **Type**: ``string``
- **Required**: No
- **Description**: Human-readable description of the protocol
- **Example**: ``"Antibody design workflow using AntiFold and IgBert"``

inputs
~~~~~~

- **Type**: ``object``
- **Required**: Yes
- **Description**: Input parameter definitions with default values
- **Constraints**: 
  - Must be an object (dictionary)
  - Keys are input parameter names
  - Values can be any YAML type (string, number, boolean, array, object)
  - Values may contain template expressions: ``${{ ... }}``
- **Example**:

  .. code-block:: yaml

     inputs:
       pdb_str: string
       heavy_chain: string
       n_samples: 20
       temperature: 1.0
       regions: ["CDR1", "CDR2", "CDR3"]
       dev: false

outputs
~~~~~~~

- **Type**: ``array``
- **Required**: No
- **Description**: Select which task outputs to log to MLflow. Defines the protocol's output interface alongside ``inputs``.
- **Constraints**: 
  - Array of output specifications
  - Each specification must reference a valid task ID
  - Fields must exist in the task's ``response_mapping``
  - Field types: ``metric`` (default), ``parameter``, ``artifact``, ``dataset``
- **Example (shorthand - defaults to metric)**:

  .. code-block:: yaml

     outputs:
       - task: igbert_score
         fields: [log_prob]  # defaults to type: metric

- **Example (explicit types)**:

  .. code-block:: yaml

     outputs:
       - task: antifold_generate
         fields:
           - name: score
             type: metric
           - name: heavy
             type: artifact
           - name: temperature
             type: parameter
           - name: sequences
             type: dataset

**Field Types:**
  - ``metric`` (default): Numeric values logged with ``mlflow.log_metric()``
  - ``parameter``: Configuration values logged with ``mlflow.log_param()``
  - ``artifact``: Files/data blobs (PDB, FASTA, etc.) logged with ``mlflow.log_artifact()``
  - ``dataset``: MLflow datasets logged with ``mlflow.log_input()``

execution
~~~~~~~~~

- **Type**: ``object``
- **Required**: No
- **Description**: Execution configuration controlling how the protocol runs
- **Properties**:
  - ``progress``: Progress tracking configuration
  - ``ranking``: Top-N ranking configuration
  - ``concurrency``: Concurrency control
  - ``writing``: Output writing configuration
- **Example**:

  .. code-block:: yaml

     execution:
       progress:
         total_expected: ${{ n_samples }}
       ranking:
         field: "log_prob"
         order: "descending"
         top_n: 10
       concurrency:
         workflow: 2
         tasks: 8
       writing:
         deduplicate: true
         max_dedupe_size: 1000

execution.progress
~~~~~~~~~~~~~~~~~~

- **Type**: ``object``
- **Required**: No
- **Description**: Progress tracking configuration
- **Properties**:
  - ``total_expected``: ``integer`` or template expression - Expected number of final records
- **Example**:

  .. code-block:: yaml

     execution:
       progress:
         total_expected: ${{ n_samples }}

execution.ranking
~~~~~~~~~~~~~~~~~

- **Type**: ``object``
- **Required**: No
- **Description**: Top-N ranking configuration for real-time updates
- **Properties**:
  - ``field``: ``string`` - Field name from ``response_mapping`` to rank by
  - ``order``: ``string`` - ``"ascending"`` or ``"descending"``
  - ``top_n``: ``integer`` - Number of top records to keep in heap
- **Example**:

  .. code-block:: yaml

     execution:
       ranking:
         field: "log_prob"
         order: "descending"
         top_n: 10

execution.concurrency
~~~~~~~~~~~~~~~~~~~~~~

- **Type**: ``object``
- **Required**: No
- **Description**: Concurrency control for workflow execution
- **Properties**:
  - ``workflow``: ``integer`` - Number of concurrent protocol instances
  - ``tasks``: ``integer`` - Per-instance max concurrent sub-tasks
- **Example**:

  .. code-block:: yaml

     execution:
       concurrency:
         workflow: 2
         tasks: 8

execution.writing
~~~~~~~~~~~~~~~~~

- **Type**: ``object``
- **Required**: No
- **Description**: Output writing configuration
- **Properties**:
  - ``deduplicate``: ``boolean`` - Enable deduplication of results
  - ``max_dedupe_size``: ``integer`` - Maximum deduplication cache size in MB
- **Example**:

  .. code-block:: yaml

     execution:
       writing:
         deduplicate: true
         max_dedupe_size: 1000

tasks
~~~~~

- **Type**: ``array``
- **Required**: Yes
- **Description**: List of workflow tasks to execute
- **Constraints**: 
  - Must be a non-empty array
  - Each item must be either a ``gather`` task or a ``model`` task
- **Example**: See task definitions below

Task Definitions
----------------

Gather Task
~~~~~~~~~~~

Gather tasks collect and batch data from previous tasks.

**Required Fields:**
  - ``id``: ``string`` - Unique task identifier
  - ``type``: ``"gather"`` - Task type identifier
  - ``from``: ``string`` - Source task ID to gather from
  - ``fields``: ``array[string]`` - Field names to gather

**Optional Fields:**
  - ``depends_on``: ``array[string]`` - Task dependencies
  - ``into``: ``integer`` or template expression - Batch size
  - ``skip_if_empty``: ``boolean`` - Skip if source is empty

**Example**:

  .. code-block:: yaml

     - id: sequences_batches_scoring
       type: gather
       from: antifold_generate
       fields: [heavy, light]
       depends_on: [antifold_generate]
       into: ${{ max_score_batch }}
       skip_if_empty: true

Model Task
~~~~~~~~~~

Model tasks execute API calls to BioLM models.

**Required Fields:**
  - ``id``: ``string`` - Unique task identifier
  - ``request_body``: ``object`` - Request payload
  - ``response_mapping``: ``object`` - Response field mappings

**Model Identification (one of two patterns):**

**Pattern 1: ``class``/``app``/``method``**
  - ``class``: ``string`` - Model class name (e.g., ``"AntiFoldModel"``)
  - ``app``: ``string`` - Application identifier (e.g., ``"antifold"``)
  - ``method``: ``string`` - Method name (e.g., ``"generate"``, ``"predict"``, ``"predict_log_prob"``)

**Pattern 2: ``slug``/``action``**
  - ``slug``: ``string`` - Model slug (e.g., ``"antifold"``, ``"igbert-paired"``)
  - ``action``: ``string`` - Action name (e.g., ``"generate"``, ``"predict"``, ``"encode"``, ``"similarity"``)

**Optional Fields:**
  - ``type``: ``string`` - Defaults to ``"task"`` if not specified
  - ``depends_on``: ``array[string]`` - Task dependencies
  - ``foreach``: template expression - Iterate over array, creating subtasks
  - ``skip_if``: template expression - Conditional execution
  - ``skip_if_empty``: ``boolean`` - Skip if dependencies are empty
  - ``fail_on_error``: ``boolean`` - Whether to fail on errors (default: ``true``)

**Request Body Structure:**

  .. code-block:: yaml

     request_body:
       items: array or template expression  # Input items
       params: object                        # Optional parameters

**Response Mapping Structure:**

  .. code-block:: yaml

     response_mapping:
       field_name: "${{ response.results[*].path.to.field }}"

**Example (Pattern 1):**

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
       fail_on_error: true

**Example (Pattern 2):**

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

Template Expressions
--------------------

Template expressions use the syntax: ``${{ expression }}``

**Examples:**
  - ``${{ n_samples }}`` - Variable reference
  - ``${{ n_samples // execution.concurrency.workflow }}`` - Expression with operators
  - ``${{ sequences_batches_scoring.results }}`` - Nested property access
  - ``${{ item.designed_pdb }}`` - Item context in foreach loops
  - ``"${{ response.results[*].log_prob }}"`` - JSONPath expression

**Validation Rules:**
  1. **Syntax Validation**: Must match pattern ``^\$\{\{.*\}\}$``
  2. **Expression Validation**: 
     - Basic syntax checking (balanced braces, valid operators)
     - Variable references should reference valid ``inputs`` or task IDs
     - JSONPath expressions in ``response_mapping`` should be valid JSONPath syntax
  3. **Type Checking**: 
     - Cannot validate types at schema validation time (requires runtime)
     - Schema validation should check syntax only

**JSONPath in Response Mapping:**

Response mapping values use JSONPath-like syntax to extract data from API responses:
  - ``"${{ response.results[*].field }}"`` - Extract field from all results
  - ``"${{ response.results[*].sequences[*].heavy }}"`` - Nested array access
  - ``"${{ response.results[0].field }}"`` - Single result access

Validation Strategy
-------------------

**Phase 1: Structural Validation (JSON Schema)**
  - Validate YAML syntax
  - Validate required fields
  - Validate data types
  - Validate enum values
  - Validate conditional requirements (oneOf patterns)

**Phase 2: Semantic Validation (Custom Python)**
  - Validate template expression syntax
  - Validate task ID references in ``depends_on``, ``from``, ``foreach``, ``outputs``
  - Validate JSONPath syntax in ``response_mapping``
  - Validate slug/action mappings (requires API lookup)
  - Validate class/app/method combinations (requires API lookup)
  - Check for circular dependencies in task graph
  - Validate that ``outputs`` references valid task IDs and fields

**Phase 3: Runtime Validation (During Execution)**
  - Validate template expression evaluation
  - Validate API request/response schemas
  - Validate actual data types at runtime

For more information about using protocols, see the :doc:`../usage/inputs`, :doc:`../usage/tasks`, :doc:`../usage/outputs`, and :doc:`../usage/settings` sections.
