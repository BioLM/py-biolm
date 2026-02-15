Protocol MLflow Logging
=======================

The protocol MLflow logging functionality allows you to log protocol execution results to MLflow based on the protocol's ``outputs`` configuration. This enables experiment tracking, model comparison, and result analysis.

Installation
------------

MLflow support is an optional dependency. Install it with:

.. code-block:: bash

   pip install biolmai[mlflow]

If MLflow is not installed, the logging functions will raise ``MLflowNotAvailableError`` with installation instructions.

Overview
--------

The MLflow logging system follows a two-stage approach:

1. **Preparation Stage**: Select and process all results according to output rules (no MLflow interaction)
2. **Logging Stage**: Log prepared data to MLflow (compartmentalized to avoid partial logs)

This ensures that if any error occurs during preparation, no partial data is logged to MLflow.

Main Function
-------------

.. py:function:: log_protocol_results(results, outputs_config, account_name, workspace_name, protocol_name, protocol_metadata=None, mlflow_uri=None, dry_run=False, aggregate_over='selected')

   Main entry point for logging protocol results to MLflow. The MLflow experiment name is built as ``{account_name}/{workspace_name}/{protocol_name}``.

   **Parameters:**

   * **results** (*Union[List[dict], str]*) – List of result dictionaries or path to JSONL file
   * **outputs_config** (*Union[List[dict], str, dict]*) – Outputs configuration (list of output rules, protocol dict, or file path)
   * **account_name** (*str*) – Account name for the experiment path
   * **workspace_name** (*str*) – Workspace name for the experiment path
   * **protocol_name** (*str*) – Protocol name/slug for the experiment path
   * **protocol_metadata** (*dict, optional*) – Protocol metadata dictionary with keys:
     * ``name``: Protocol name
     * ``version``: Protocol version
     * ``inputs``: Input parameters dictionary
   * **mlflow_uri** (*str, optional*) – MLflow tracking URI (default: ``"https://mlflow.biolm.ai/"``)
   * **dry_run** (*bool*) – If True, prepare data but don't log to MLflow (default: False)
   * **aggregate_over** (*str*) – Compute aggregates over ``"selected"`` or ``"all"`` rows (default: ``"selected"``)

   **Returns:**

   Dictionary with logging results:
   
   * ``dry_run``: Whether this was a dry run
   * ``experiment_name``: MLflow experiment name
   * ``parent_run_id``: Parent run ID (None if dry_run)
   * ``child_run_ids``: List of child run IDs
   * ``num_results``: Number of results processed
   * ``num_selected``: Number of results selected for logging
   * ``num_aggregates``: Number of aggregate metrics computed

   **Raises:**

   * **MLflowNotAvailableError** – If MLflow is not installed
   * **FileNotFoundError** – If results or outputs config file not found
   * **ValueError** – If configuration is invalid or evaluation fails

   **Example:**

   .. code-block:: python

      from biolmai.protocols_mlflow import log_protocol_results

      # Log results from JSONL file (experiment: account/workspace/protocol)
      result = log_protocol_results(
          results="results.jsonl",
          outputs_config="protocol.yaml",
          account_name="biolm",
          workspace_name="workspace1",
          protocol_name="my_protocol",
          protocol_metadata={
              "name": "My Protocol",
              "version": "1.0",
              "inputs": {"n_samples": 20, "temperature": 1.0}
          }
      )

      print(f"Logged {result['num_selected']} results to MLflow")
      print(f"Parent run ID: {result['parent_run_id']}")

Input Formats
~~~~~~~~~~~~~

Both ``results`` and ``outputs_config`` accept either file paths or in-memory data,
giving you flexibility in how you pass data to ``log_protocol_results``.

**results** – One of:

* **File path** (str): Path to a JSONL file. Each line must be a single JSON object (one result record per line). Supports compressed formats:
  * ``.jsonl`` – plain text
  * ``.jsonl.gz`` – gzip compressed
  * ``.zip`` – zip archive containing a ``.jsonl`` file (e.g. ``results_{id}.jsonl.zip``)
* **List of dicts**: In-memory list of result dictionaries. Use this when you have results in memory (e.g. from a protocol run, loaded from a database, or built programmatically).

**outputs_config** – One of:

* **File path** (str): Path to a YAML file. Can be:
  * A list of output rules (the ``outputs`` section content)
  * A full protocol YAML (the ``outputs`` section will be extracted automatically)
* **List of dicts**: In-memory list of output rule dictionaries.
* **Dict**: A protocol dictionary with an ``outputs`` key (extracted automatically), or a single output rule.

**Examples by format:**

.. code-block:: python

   # File paths (most common)
   log_protocol_results(
       results="results.jsonl",
       outputs_config="protocol.yaml",
       account_name="acme", workspace_name="lab", protocol_name="my_protocol"
   )

   # In-memory results, outputs from file
   log_protocol_results(
       results=[{"score": 0.8, "sequence": "ACDE..."}, ...],
       outputs_config="protocol.yaml",
       account_name="acme", workspace_name="lab", protocol_name="my_protocol"
   )

   # Both in-memory (no files needed)
   log_protocol_results(
       results=[{"score": 0.8}, {"score": 0.6}],
       outputs_config=[{"log": {"metrics": {"score": "${{ score }}"}}}],
       account_name="acme", workspace_name="lab", protocol_name="my_protocol"
   )

Helper Functions
----------------

.. py:function:: load_results(results)

   Load results from a list or JSONL file. Supports plain (``.jsonl``), gzip
   (``.jsonl.gz``), and zip (``.zip`` with a ``.jsonl`` file inside) formats.

   **Parameters:**

   * **results** (*Union[List[dict], str]*) – List of dicts or path to JSONL file (optionally compressed)

   **Returns:**

   List of result dictionaries

.. py:function:: load_outputs_config(outputs_config)

   Load outputs configuration from various formats.

   **Parameters:**

   * **outputs_config** (*Union[List[dict], str, dict]*) – Can be:
     * List of output rule dicts
     * Path to YAML file containing outputs config
     * Path to protocol YAML file (will extract outputs section)
     * Protocol dict (will extract outputs section)

   **Returns:**

   List of output rule dictionaries

.. py:function:: prepare_logging_data(results, outputs_config, protocol_metadata=None, aggregate_over='selected')

   Prepare all logging data (Stage 1: No MLflow interaction).

   **Parameters:**

   * **results** (*List[dict]*) – List of result dictionaries
   * **outputs_config** (*List[dict]*) – List of output rule configurations
   * **protocol_metadata** (*dict, optional*) – Protocol metadata
   * **aggregate_over** (*str*) – ``"selected"`` or ``"all"`` for aggregate computation

   **Returns:**

   Dictionary with prepared logging data

.. py:function:: log_to_mlflow(prepared_data, experiment_name, mlflow_uri=None, dry_run=False)

   Log prepared data to MLflow (Stage 2: MLflow interaction only).

   **Parameters:**

   * **prepared_data** (*dict*) – Prepared logging data from ``prepare_logging_data()``
   * **experiment_name** (*str*) – MLflow experiment name
   * **mlflow_uri** (*str, optional*) – MLflow tracking URI
   * **dry_run** (*bool*) – If True, don't actually log to MLflow

   **Returns:**

   Dictionary with run IDs and status

Usage Examples
--------------

Basic Logging
~~~~~~~~~~~~~

.. code-block:: python

   from biolmai.protocols_mlflow import log_protocol_results

   # Simple logging with protocol file (experiment: account/workspace/protocol)
   result = log_protocol_results(
       results="results.jsonl",
       outputs_config="protocol.yaml",
       account_name="my_account",
       workspace_name="my_workspace",
       protocol_name="my_protocol"
   )

Dry Run
~~~~~~~

.. code-block:: python

   # Prepare data without logging to MLflow
   result = log_protocol_results(
       results="results.jsonl",
       outputs_config="protocol.yaml",
       account_name="my_account",
       workspace_name="my_workspace",
       protocol_name="my_protocol",
       dry_run=True
   )

   print(f"Would log {result['num_selected']} results")

Custom MLflow URI
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use custom MLflow server
   result = log_protocol_results(
       results="results.jsonl",
       outputs_config="protocol.yaml",
       account_name="my_account",
       workspace_name="my_workspace",
       protocol_name="my_protocol",
       mlflow_uri="http://localhost:5000"
   )

Programmatic Results
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use results as list instead of file
   results = [
       {"score": 0.8, "log_prob": -1.5, "sequence": "ACDEFGHIKLMNPQRSTVWY"},
       {"score": 0.6, "log_prob": -2.0, "sequence": "DEFGHIKLMNPQRSTVWY"},
   ]

   outputs_config = [
       {
           "where": "${{ score > 0.5 }}",
           "limit": 10,
           "log": {
               "metrics": {"score": "${{ score }}"},
               "params": {"temperature": "${{ log_prob }}"},
           }
       }
   ]

   result = log_protocol_results(
       results=results,
       outputs_config=outputs_config,
       account_name="my_account",
       workspace_name="my_workspace",
       protocol_name="my_protocol"
   )

Error Handling
--------------

The logging system is designed to fail fast and avoid partial logs:

* **Preparation errors**: Raised immediately with clear error messages
* **MLflow logging errors**: Fail fast, no partial logs
* **Missing fields**: Handled gracefully (return None or skip)
* **Invalid expressions**: Raise with context (which expression, which row)

If MLflow is not installed, all functions will raise ``MLflowNotAvailableError``:

.. code-block:: python

   from biolmai.protocols_mlflow import MLflowNotAvailableError, log_protocol_results

   try:
       log_protocol_results(...)
   except MLflowNotAvailableError:
       print("Install MLflow support: pip install biolmai[mlflow]")

MLflow Run Structure
--------------------

The logging system creates a hierarchical run structure:

* **Parent Run**: One per protocol execution
  * Tag: ``type: protocol``
  * Contains: Protocol metadata, aggregate metrics, ``results.jsonl`` artifact (full results)
  * Name: Based on experiment name

* **Child Runs**: One per selected result
  * Tag: ``type: model``
  * Contains: Result-specific parameters, metrics, tags, artifacts
  * Nested under parent run

Artifacts
---------

**Parent run:** The full results table is always logged as ``results.jsonl`` (one JSON object per line) to the parent (protocol) run. This provides a complete record of all protocol execution results.

**Child runs:** The system automatically generates ``sequence.json`` artifacts in seqparse format for results that contain a ``sequence`` field. Additional artifacts can be specified in the outputs configuration.

Sequence Format (seqparse)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The seqparse format is a JSON structure:

.. code-block:: json

   {
     "entries": [
       {
         "sequence": "ACDEFGHIKLMNPQRSTVWY",
         "id": "seq1",
         "metadata": {
           "score": 0.8,
           "log_prob": -1.5
         }
       }
     ]
   }

CLI Command
-----------

See :doc:`../cli/protocol` for CLI command documentation.

Related Documentation
--------------------

* :doc:`../protocols/output` - Protocol outputs configuration
* :doc:`../protocols/yaml-reference/outputs` - Outputs YAML reference


