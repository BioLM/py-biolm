``execution``
=============

The ``execution`` field controls how the protocol runs, including progress tracking, ranking, concurrency, and output writing. This field is **optional** - if omitted, default execution settings are used.

Overview
--------

Execution configuration allows you to optimize protocol performance and behavior:

- **progress**: Track execution progress for monitoring
- **ranking**: Maintain top-N results in real-time
- **concurrency**: Control parallel execution for performance
- **writing**: Configure result deduplication and caching

All execution sub-fields are optional. You can include only the ones you need.

Schema Definition
-----------------

.. jsonschema:: ../../schema/protocol_schema.json
   #/properties/execution

Progress Configuration
----------------------

Progress tracking helps monitor long-running protocols. Set ``total_expected`` to the number of final records you expect, which enables progress percentage calculations.

.. jsonschema:: ../../schema/protocol_schema.json
   #/$defs/Progress

**When to use**: Enable progress tracking when:
- Protocols run for a long time
- You want to monitor completion percentage
- You need to estimate remaining time

**Example:**

.. code-block:: yaml

   execution:
     progress:
       total_expected: ${{ n_samples }}  # Expect 20 final records

**How it works**: The protocol tracks how many records have been processed and compares against ``total_expected`` to calculate progress. This is especially useful for protocols with variable output counts.

Ranking Configuration
---------------------

Ranking maintains a heap of the top-N results during execution, enabling real-time updates of the best results even before all tasks complete.

.. jsonschema:: ../../schema/protocol_schema.json
   #/$defs/Ranking

**When to use**: Enable ranking when:
- You only care about the best N results
- You want to see top results as they're generated
- You're doing large-scale searches or optimization

**Example:**

.. code-block:: yaml

   execution:
     ranking:
       field: "log_prob"        # Field from response_mapping to rank by
       order: "descending"       # "ascending" or "descending"
       top_n: 10                 # Keep top 10 in heap

**How it works**: As results come in, they're compared against the current top-N heap. Only the best N results are kept in memory, which is efficient for large result sets.

**Important**: The ``field`` must exist in a task's ``response_mapping``. The ranking happens across all results from all tasks.

Concurrency Configuration
-------------------------

Concurrency controls how many tasks run in parallel. This is crucial for performance optimization.

.. jsonschema:: ../../schema/protocol_schema.json
   #/$defs/Concurrency

**workflow** (integer or expression)
  Number of concurrent protocol instances. Each instance processes a portion of the work.
  
  - Higher values = more parallelism but more resource usage
  - Lower values = less parallelism but more predictable resource usage
  - Can use expressions: ``${{ n_samples // 10 }}``

**tasks** (integer or expression)
  Per-instance maximum concurrent sub-tasks. This controls parallelism within each workflow instance.
  
  - Higher values = faster task execution but more API load
  - Lower values = slower but more controlled API usage
  - Consider API rate limits when setting this

**Example:**

.. code-block:: yaml

   execution:
     concurrency:
       workflow: 2   # Run 2 protocol instances in parallel
       tasks: 8       # Each instance can run up to 8 tasks concurrently

**Performance tips**:
- Start with conservative values (workflow: 1-2, tasks: 4-8)
- Increase gradually while monitoring API rate limits
- Use expressions to scale based on input size: ``${{ n_samples // 10 }}``

Writing Configuration
---------------------

Writing configuration controls how results are written and deduplicated.

.. jsonschema:: ../../schema/protocol_schema.json
   #/$defs/Writing

**deduplicate** (boolean)
  Enable deduplication of results. When enabled, duplicate results (based on content) are filtered out.
  
  - ``true``: Remove duplicate results
  - ``false``: Keep all results, including duplicates

**max_dedupe_size** (integer or expression)
  Maximum size of the deduplication cache in megabytes. When the cache exceeds this size, older entries are evicted.
  
  - Higher values = better deduplication but more memory usage
  - Lower values = less memory but potentially less effective deduplication
  - Default behavior if not specified

**Example:**

.. code-block:: yaml

   execution:
     writing:
       deduplicate: true
       max_dedupe_size: 1000  # 1 GB cache

**When to use deduplication**:
- Protocols that may generate duplicate results
- When you want unique results only
- When memory usage is acceptable

**When to disable deduplication**:
- When duplicates are meaningful
- When memory is constrained
- When you need maximum performance

Complete Example
----------------

A complete execution configuration combining all options:

.. code-block:: yaml

   execution:
     progress:
       total_expected: ${{ n_samples }}  # Track progress
     ranking:
       field: "log_prob"
       order: "descending"
       top_n: 10                          # Keep top 10
     concurrency:
       workflow: ${{ n_samples // 10 }}   # Scale with input
       tasks: 8                            # Fixed task concurrency
     writing:
       deduplicate: true
       max_dedupe_size: 1000               # 1 GB cache

Performance Considerations
--------------------------

**Concurrency tuning**:
- Start low and increase gradually
- Monitor API rate limits and errors
- Use expressions to scale with input size

**Memory management**:
- ``max_dedupe_size`` affects memory usage
- ``top_n`` ranking also uses memory (smaller is better)
- Consider your available resources

**Progress tracking**:
- ``total_expected`` should be accurate for meaningful progress
- Can use expressions: ``${{ n_samples * 2 }}`` if you expect 2x output

Common Patterns
---------------

**High-throughput protocol**:

.. code-block:: yaml

   execution:
     concurrency:
       workflow: 4
       tasks: 16
     writing:
       deduplicate: true

**Interactive/exploratory protocol**:

.. code-block:: yaml

   execution:
     ranking:
       field: "score"
       order: "descending"
       top_n: 5
     concurrency:
       workflow: 1
       tasks: 4

**Memory-constrained protocol**:

.. code-block:: yaml

   execution:
     writing:
       deduplicate: false
     ranking:
