``outputs``
===========

The ``outputs`` field defines output rules for MLflow logging. Each rule operates on the final merged results table (a single list of records from all tasks), applies filtering/ordering, and specifies what to log (params, metrics, tags, aggregates, artifacts).

Overview
--------

Output rules are the **interface** between your protocol execution and MLflow tracking. They allow you to:

- **Filter** results using row-level expressions
- **Order** results by specific fields
- **Limit** the number of rows processed
- **Log** parameters, metrics, tags, aggregates, and artifacts to MLflow

**When to use**: Include ``outputs`` when you want to track protocol runs in MLflow for:
- Experiment tracking
- Model comparison
- Reproducibility
- Result analysis

If you don't need MLflow logging, you can omit the ``outputs`` field entirely.

**Important**: Protocols always output a single merged list of records from all tasks. The ``outputs`` field operates on this final merged results table, not on individual task outputs.

How Output Rules Work
---------------------

Output rules process results in this order:

1. **Filter**: Apply ``where`` expression to filter rows from the merged results
2. **Order**: Sort results by ``order_by`` specification
3. **Limit**: Take top N rows (via ``limit``)
4. **Log**: Extract and log specified fields to MLflow

Each rule operates independently - you can have multiple rules logging different aspects of the merged results.

Output Rule Structure
---------------------

Each output rule specifies:

- **where**: Optional row-level filter expression
- **order_by**: Optional ordering specification
- **limit**: Maximum rows to select (default: 200)
- **run**: Optional MLflow run configuration
- **log**: What to log (params, metrics, tags, aggregates, artifacts)

**Output Rule Properties**:

**where** (string, optional)
  Row-level filter expression. Only rows where this expression evaluates to ``true`` are included. Uses template expression syntax.
  
  Examples:
  - ``"${{ score > 0.5 }}"`` - Only rows with score > 0.5
  - ``"${{ log_prob > -2.0 and temperature < 1.5 }}"`` - Complex condition

**order_by** (array, optional)
  Sort specification. Results are sorted by the first field, then by the second, etc.
  
  Each entry has:
  - ``field``: Field name to sort by (must exist in the merged results)
  - ``order``: ``"asc"`` or ``"desc"``

**limit** (integer, optional, default: 200)
  Maximum number of rows to process. After filtering and ordering, only the top N rows are kept.

**run** (object, optional)
  MLflow run configuration. Allows creating child runs or specifying run metadata.

**log** (object, optional)
  What to log to MLflow. See Log Specification below.

Log Specification
-------------------

The ``log`` field defines what to log to MLflow. All fields are optional - include only what you need.

**Log Specification Properties**:

**params** (object, optional)
  Parameters to log. Each key becomes a parameter name, each value is a template expression.
  
  Example:
  .. code-block:: yaml
  
     params:
       temperature: "${{ temperature }}"
       n_samples: "${{ n_samples }}"

**metrics** (object, optional)
  Metrics to log. Each key becomes a metric name, each value is a template expression that should evaluate to a number.
  
  Example:
  .. code-block:: yaml
  
     metrics:
       mean_score: "${{ score }}"
       max_log_prob: "${{ log_prob }}"

**tags** (object, optional)
  Tags to log. Each key becomes a tag name, each value is a template expression (usually a string).
  
  Example:
  .. code-block:: yaml
  
     tags:
       model_version: "${{ model_version }}"
       protocol_name: "antibody_design"

**aggregates** (array, optional)
  Compute aggregate statistics over the selected rows. See Aggregate Specification.

**artifacts** (array, optional)
  Log files and data artifacts. See Artifact Specification.

Aggregate Specification
-----------------------

Aggregates compute statistics over the selected rows. Useful for summarizing results.

**Example:**

.. code-block:: yaml

   aggregates:
     - field: score
       ops: [mean, max, p95]
     - field: __rows__
       ops: [count]

**Fields:**

**field** (required)
  Field name to aggregate over. Must exist in the merged results. Use ``"__rows__"`` to count rows.

**ops** (array, required)
  Operations to compute. Supported operations:
  - ``mean``: Arithmetic mean
  - ``median``: Median value
  - ``std``: Standard deviation
  - ``min``: Minimum value
  - ``max``: Maximum value
  - ``p95``: 95th percentile
  - ``p99``: 99th percentile
  - ``count``: Number of non-null values

**Example:**

.. code-block:: yaml

   aggregates:
     - field: score
       ops: [mean, max, p95]
     - field: log_prob
       ops: [mean, std]

This computes mean, max, and p95 for ``score``, and mean and std for ``log_prob``, logging them as metrics.

Artifact Specification
----------------------

Artifacts define files and data to log to MLflow. Supports various artifact types.

**Example:**

.. code-block:: yaml

   artifacts:
     - type: pdb
       name: "best_structure.pdb"
       content: "${{ designed_pdb }}"
     - type: fasta
       name: "sequences.fasta"
       entries:
         - id: "seq_1"
           sequence: "${{ heavy }}"

**Fields:**

**type** (required)
  Artifact type. Common types:
  - ``pdb``: Protein structure files
  - ``fasta``: Sequence files
  - ``json``: JSON data files
  - ``csv``: CSV data files
  - ``text``: Plain text files

**name** (string, required)
  Artifact name/path in MLflow.

**content** (string, optional)
  Direct content for simple artifacts (e.g., single PDB string).

**entries** (array, optional)
  For sequence-style artifacts (like FASTA), list of sequence entries.

**Example - Simple artifact:**

.. code-block:: yaml

   artifacts:
     - type: pdb
       name: "best_structure.pdb"
       content: "${{ designed_pdb }}"

**Example - Sequence artifact:**

.. code-block:: yaml

   artifacts:
     - type: fasta
       name: "sequences.fasta"
       entries:
         - id: "seq_1"
           sequence: "${{ heavy }}"
           metadata:
             score: "${{ score }}"

Sequence Entry
--------------

Sequence entries are used in sequence-style artifacts (like FASTA files).

**Example:**

.. code-block:: yaml

   entries:
     - id: "seq_1"
       sequence: "${{ heavy }}"
       metadata:
         chain: "H"

**Fields:**

**id** (optional)
  Sequence identifier.

**sequence** (string, required)
  Sequence content (template expression).

**metadata** (object, optional)
  Additional metadata to include with the sequence.

Examples
--------

Basic Output Rule
~~~~~~~~~~~~~~~~~

Log a simple metric from the merged results:

.. code-block:: yaml

   outputs:
     - limit: 100
       log:
         metrics:
           mean_log_prob: "${{ log_prob }}"

Filtered and Ordered Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filter results, order by score, and log multiple things:

.. code-block:: yaml

   outputs:
     - where: "${{ score > 0.5 }}"           # Only high-scoring results
       order_by:
         - field: log_prob
           order: desc                       # Best first
       limit: 50                              # Top 50
       log:
         params:
           temperature: "${{ temperature }}"
         metrics:
           max_score: "${{ score }}"
           best_log_prob: "${{ log_prob }}"
         aggregates:
           - field: score
             ops: [mean, max, p95]            # Summary statistics

Multiple Output Rules
~~~~~~~~~~~~~~~~~~~~~

Log different aspects with different filters:

.. code-block:: yaml

   outputs:
     - where: "${{ log_prob > -2.0 }}"
       log:
         metrics:
           mean_log_prob: "${{ log_prob }}"
     
     - where: "${{ plddt > 0.8 }}"
       log:
         metrics:
           mean_plddt: "${{ plddt }}"
         artifacts:
           - type: pdb
             name: "structures.pdb"
             content: "${{ designed_pdb }}"

Artifact Logging
~~~~~~~~~~~~~~~~

Log files and sequences:

.. code-block:: yaml

   outputs:
     - where: "${{ plddt > 0.8 }}"
       log:
         artifacts:
           - type: pdb
             name: "best_structure.pdb"
             content: "${{ designed_pdb }}"
           - type: fasta
             name: "sequences.fasta"
             entries:
               - id: "heavy_chain"
                 sequence: "${{ heavy }}"
                 metadata:
                   score: "${{ score }}"
                   log_prob: "${{ log_prob }}"
               - id: "light_chain"
                 sequence: "${{ light }}"
                 metadata:
                   score: "${{ score }}"

Complete Example
~~~~~~~~~~~~~~~~

Comprehensive output configuration:

.. code-block:: yaml

   outputs:
     - where: "${{ log_prob > -2.0 }}"
       order_by:
         - field: log_prob
           order: desc
       limit: 100
       log:
         params:
           n_samples: "${{ n_samples }}"
           temperature: "${{ temperature }}"
         metrics:
           max_log_prob: "${{ log_prob }}"
         tags:
           protocol: "antibody_design"
         aggregates:
           - field: log_prob
             ops: [mean, std, p95]
     
     - where: "${{ plddt > 0.8 }}"
       limit: 10
       log:
         artifacts:
           - type: pdb
             name: "top_structures.pdb"
             content: "${{ designed_pdb }}"

Best Practices
--------------

1. **Use filtering**: Filter to only log relevant results (saves space and improves clarity)
2. **Order before limiting**: Use ``order_by`` to ensure you get the best results
3. **Set appropriate limits**: Don't log more than you need (default 200 is often too high)
4. **Use aggregates**: Summarize large result sets with aggregates
5. **Name artifacts clearly**: Use descriptive names for artifacts
6. **Log parameters**: Include key protocol parameters for reproducibility

Common Patterns
---------------

**Top-N results pattern**:

.. code-block:: yaml

   outputs:
     - order_by:
         - field: score
           order: desc
       limit: 10  # Top 10 only
       log:
         metrics:
           top_score: "${{ score }}"

**Summary statistics pattern**:

.. code-block:: yaml

   outputs:
     - log:
         aggregates:
           - field: score
             ops: [mean, std, min, max, p95]

**Multiple filtered views pattern**:

.. code-block:: yaml

   outputs:
     - where: "${{ score > 0.9 }}"
       log:
         metrics:
           high_score_count: "${{ __rows__ }}"
     - where: "${{ log_prob > -1.0 }}"
       log:
         metrics:
           high_log_prob_mean: "${{ log_prob }}"
