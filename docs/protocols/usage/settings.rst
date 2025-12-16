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

Schema Definition
-----------------

.. jsonschema:: ../../../schema/protocol_schema.json
   :pointer: /properties/execution

Progress Configuration
----------------------

Progress tracking configuration for monitoring protocol execution.

.. jsonschema:: ../../../schema/protocol_schema.json
   :pointer: /$defs/Progress

**Example:**

.. code-block:: yaml

   execution:
     progress:
       total_expected: ${{ n_samples }}

Ranking Configuration
---------------------

Top-N ranking configuration for real-time updates during execution.

.. jsonschema:: ../../../schema/protocol_schema.json
   :pointer: /$defs/Ranking

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

.. jsonschema:: ../../../schema/protocol_schema.json
   :pointer: /$defs/Concurrency

**Example:**

.. code-block:: yaml

   execution:
     concurrency:
       workflow: 2  # Number of concurrent protocol instances
       tasks: 8     # Per-instance max concurrent sub-tasks

Writing Configuration
---------------------

Output writing configuration for result handling.

.. jsonschema:: ../../../schema/protocol_schema.json
   :pointer: /$defs/Writing

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
       max_dedupe_size: 1000
