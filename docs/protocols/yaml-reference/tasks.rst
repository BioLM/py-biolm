Task Definitions
================

Tasks define the workflow steps in a Protocol. There are two types of tasks: Gather tasks and Model (API) tasks.

Gather Tasks
------------

Gather tasks collect and batch data from previous tasks.

**Properties**:

- **id** (string, required): Unique task identifier
- **type** (string, required): Must be ``"gather"``
- **from** (string, required): Source task ID to gather from
- **fields** (array of strings, required): Field names to collect (minimum 1 item)
- **into** (integer or expression, optional): Batch size (minimum 1)
- **depends_on** (array of strings, optional): Task dependencies
- **skip_if_empty** (boolean, optional): Skip if dependencies are empty
- **skip_if** (string or expression, optional): Conditional execution expression
- **foreach** (string or expression, optional): Iterate over array, creating subtasks
- **response_mapping** (object, optional): Maps API response to protocol fields

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
- **response_mapping** (object, optional): Maps API response to protocol fields
- **One of** (required):
  - **slug** (string or expression) + **action** (string: ``"predict"``, ``"encode"``, ``"generate"``, ``"similarity"``)
  - **class** (string) + **app** (string) + **method** (string)
- **depends_on** (array of strings, optional): Task dependencies
- **foreach** (string or expression, optional): Iterate over array, creating subtasks
- **skip_if** (string or expression, optional): Conditional execution expression
- **skip_if_empty** (boolean, optional): Skip if dependencies are empty
- **fail_on_error** (boolean, optional): Whether to fail on errors (default: ``true``)
- **type** (string, optional): Defaults to ``"task"``

Request Body
------------

**Properties**:

- **items** (array or expression, required): Input items to process
- **params** (object, optional): Additional parameters for the API call

Response Mapping
----------------

**Properties**:

- Object with string keys and JSONPath expression values
- Each key becomes a field name available to downstream tasks
- Each value is a JSONPath expression: ``"${{ response.path.to.field }}"``
- Supports array wildcards: ``"${{ response.results[*].field }}"``
