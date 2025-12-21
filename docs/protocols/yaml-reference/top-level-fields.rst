Top-Level Fields
================

This section documents the top-level fields of a Protocol YAML file.

name
----

Protocol name identifier.

**Example:**

.. code-block:: yaml

   name: "antibody_design_v1"

**What it is:** A string that uniquely identifies your protocol. Use lowercase letters, numbers, and underscores.

description
-----------

Human-readable description of what the protocol does.

**Example:**

.. code-block:: yaml

   description: "Designs antibody sequences using AntiFold and scores them with IgBERT"

**What it is:** Optional string that describes your protocol. Useful for documentation and cataloging.

about
-----

Informational fields for cataloging and citation.

**Example:**

.. code-block:: yaml

   about:
     title: "Antibody Design Protocol"
     authors: ["Jane Doe", "John Smith"]
     version: "1.0.0"
     description: "Protocol for designing and scoring antibody sequences"

**What it is:** Optional object with metadata about the protocol. Can include fields like ``title``, ``authors``, ``version``, ``description``, etc.

schema_version
--------------

Schema version for validation.

**Example:**

.. code-block:: yaml

   schema_version: 1

**What it is:** Integer or expression specifying which schema version to use for validation. Defaults to 1 if omitted. Currently only version 1 is supported.

protocol_version
----------------

Protocol version identifier.

**Example:**

.. code-block:: yaml

   protocol_version: "1.0.0"

**What it is:** Optional string to track protocol evolution. Use semantic versioning (e.g., "1.0.0", "1.1.0") or any versioning scheme you prefer.

inputs
------

Input parameters with default values.

**Example:**

.. code-block:: yaml

   inputs:
     pdb_str: string
     heavy_chain: "A"
     n_samples: 20
     temperature: 1.0
     regions: ["CDR1", "CDR2", "CDR3"]

**What it is:** Object (dictionary) where keys are input parameter names and values are default values. Values can be any YAML type (string, number, boolean, array, object) and may contain template expressions like ``${{ n_samples // 10 }}``.

**How to use:** Reference inputs in your protocol using ``${{ input_name }}``. Inputs can be overridden when executing the protocol.

outputs
-------

Output rules for MLflow logging.

**Example:**

.. code-block:: yaml

   outputs:
     - where: "${{ log_prob > -2.0 }}"
       limit: 100
       log:
         metrics:
           mean_log_prob: "${{ log_prob }}"

**What it is:** Array of output rule objects. Each rule operates on the final merged results table (a single list of records from all tasks), applies filtering/ordering, and specifies what to log to MLflow. See the :doc:`outputs` section for details.

**When to use:** Include this if you want to track protocol runs in MLflow. Omit if you don't need MLflow logging.

execution
---------

Execution configuration (progress tracking, ranking, concurrency, output writing).

**Example:**

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

**What it is:** Optional object that controls how the protocol runs. See the :doc:`execution` section for details on each sub-field.

**Sub-fields:**
- ``progress``: Track execution progress
- ``ranking``: Keep top-N results
- ``concurrency``: Control parallel execution
- ``writing``: Configure result deduplication

tasks
-----

List of workflow tasks to execute.

**Example:**

.. code-block:: yaml

   tasks:
     - id: generate
       slug: antifold
       action: generate
       request_body:
         items: [{pdb: ${{ pdb_str }}}]
       response_mapping:
         sequences: "${{ response.results[*].sequences }}"
     
     - id: score
       slug: igbert-paired
       action: predict
       depends_on: [generate]
       request_body:
         items: ${{ generate.results }}
       response_mapping:
         log_prob: "${{ response.results[*].log_prob }}"

**What it is:** Array of task objects. Tasks define the workflow steps. Each task can be either a gather task or a model (API) task. Tasks execute based on their dependencies, forming a directed acyclic graph (DAG). See the :doc:`tasks` section for details.

**Required:** This field is required. You must define at least one task.
