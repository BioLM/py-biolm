``settings`` (execution)
========================

The ``execution`` field (also referred to as "settings") controls how the protocol runs, including progress tracking, ranking, concurrency, and output writing.

Overview
--------

Execution configuration includes:
- **progress**: Progress tracking configuration
- **ranking**: Top-N ranking configuration for real-time updates
- **concurrency**: Concurrency control for workflow execution
- **writing**: Output writing configuration

Progress Configuration
----------------------

Progress tracking configuration for monitoring protocol execution.

**Properties**:
- **total_expected** (integer or expression, optional): Expected number of final records

**Example:**

.. code-block:: yaml

   execution:
     progress:
       total_expected: ${{ n_samples }}

Ranking Configuration
---------------------

Top-N ranking configuration for real-time updates during execution.

**Properties**:

**field** (string, required)
  Field name from ``response_mapping`` to rank by. Must exist in the task's response mapping.

**order** (string, required)
  Sort order: ``"ascending"`` or ``"descending"``.

**top_n** (integer or expression, required)
  Number of top results to maintain. Must be at least 1.

**Example:**

.. code-block:: yaml

   execution:
     ranking:
       field: "log_prob"
       order: "descending"
       top_n: 10

Concurrency Configuration
--------------------------

Concurrency control for workflow execution.

**Properties**:
- **workflow** (integer or expression, required): Number of concurrent protocol instances
- **tasks** (integer or expression, required): Per-instance max concurrent sub-tasks

**Example:**

.. code-block:: yaml

   execution:
     concurrency:
       workflow: 2  # Number of concurrent protocol instances
       tasks: 8     # Per-instance max concurrent sub-tasks

Writing Configuration
---------------------

Output writing configuration for result handling.

**Properties**:
- **deduplicate** (boolean, optional): Enable deduplication of results
- **max_dedupe_size** (integer or expression, optional): Maximum deduplication cache size in MB

**Example:**

.. code-block:: yaml

   execution:
     writing:
       deduplicate: true
       max_dedupe_size: 1000  # Maximum deduplication cache size in MB

Complete Example
----------------

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
