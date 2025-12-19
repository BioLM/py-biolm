Execution Configuration
=======================

Execution configuration controls how the protocol runs, including progress tracking, ranking, concurrency, and output writing.

execution.progress
------------------

Track execution progress for monitoring.

**Example:**

.. code-block:: yaml

   execution:
     progress:
       total_expected: ${{ n_samples }}

**Fields:**
- ``total_expected``: Number of final records you expect. Used to calculate progress percentage. Can be a number or expression like ``${{ n_samples }}``.

execution.ranking
-----------------

Keep only the top N results during execution, sorted by a field value.

**Example:**

.. code-block:: yaml

   execution:
     ranking:
       field: "log_prob"
       order: "descending"
       top_n: 10

**Fields:**
- ``field``: Name of a field from your task's ``response_mapping`` to use for ranking (e.g., ``"log_prob"``, ``"score"``)
- ``order``: How to sort - ``"ascending"`` (lower is better) or ``"descending"`` (higher is better)
- ``top_n``: How many top results to keep (must be at least 1). Can be a number or expression.

execution.concurrency
---------------------

Control how many tasks run in parallel.

**Example:**

.. code-block:: yaml

   execution:
     concurrency:
       workflow: 2
       tasks: 8

**Fields:**
- ``workflow``: Number of protocol instances to run in parallel (must be at least 1)
- ``tasks``: Maximum number of concurrent sub-tasks per instance (must be at least 1)

execution.writing
-----------------

Configure how results are written and deduplicated.

**Example:**

.. code-block:: yaml

   execution:
     writing:
       deduplicate: true
       max_dedupe_size: 1000

**Fields:**
- ``deduplicate``: Set to ``true`` to remove duplicate results, ``false`` to keep all results
- ``max_dedupe_size``: Maximum size of deduplication cache in megabytes (must be at least 1)
