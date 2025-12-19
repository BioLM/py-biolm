``outputs``
===========

The ``outputs`` field defines output rules for MLflow logging. Each rule operates on the final merged results table (a single list of records from all tasks), applies filtering/ordering, and specifies what to log (params, metrics, tags, aggregates, artifacts).

Overview
--------

Output rules allow you to:
- **Filter** results using row-level expressions
- **Order** results by specific fields
- **Limit** the number of rows processed
- **Log** parameters, metrics, tags, aggregates, and artifacts to MLflow

**Important**: Protocols always output a single merged list of records from all tasks. The ``outputs`` field operates on this final merged results table, not on individual task outputs.

Output Rule Structure
---------------------

Each output rule specifies:

- **where**: Optional row-level filter expression
- **order_by**: Optional ordering specification
- **limit**: Maximum rows to select (default: 200)
- **run**: Optional MLflow run configuration
- **log**: What to log (params, metrics, tags, aggregates, artifacts)

**Output Rule Properties**:

- **where** (string or expression, optional): Row-level filter expression
- **order_by** (array, optional): Ordering specification (array of objects with ``field`` and ``order``: ``"asc"`` or ``"desc"``)
- **limit** (integer or expression, optional): Maximum number of rows selected (default: 200)
- **run** (object, optional): MLflow run configuration
  - **name** (string or expression, optional): MLflow run name
- **log** (object, optional): What to log to MLflow (see Log Specification below)

Log Specification
----------------

The ``log`` field defines what to log to MLflow:

**Log Specification Properties**:

- **params** (object, optional): Parameters to log (key-value mapping, values can be scalars or expressions)
- **metrics** (object, optional): Metrics to log (key-value mapping, values can be scalars or expressions)
- **tags** (object, optional): Tags to log (key-value mapping, values can be scalars or expressions)
- **aggregates** (array, optional): Parent-run aggregates over selected rows (see Aggregate Specification below)
- **artifacts** (array, optional): Artifacts to log (see Artifact Specification below)

Aggregate Specification
------------------------

Aggregates compute statistics over selected rows:

**Aggregate Specification Properties**:

- **field** (string, required): Field name to aggregate over (or ``"__rows__"`` for row count)
- **ops** (array of strings, required): Statistical operations to perform
  - Allowed values: ``"count"``, ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"p50"``, ``"p90"``, ``"p95"``, ``"p99"``, ``"std"``

Artifact Specification
-----------------------

Artifacts define files and data to log:

**Artifact Specification Properties**:

- **type** (string, required): Artifact type - one of: ``"seqparse"``, ``"pdb"``, ``"fasta"``, ``"table"``, ``"msa"``, ``"plot"``, ``"json"``, ``"text"``
- **name** (string, optional): Artifact name
- **path** (string, optional): File path for artifact
- **content** (string, object, or expression, optional): Text or JSON payload (for pdb, json, text types)
- **entries** (array, optional): Sequence-style entries (for seqparse, fasta types)
- **rows** (array or expression, optional): Row collection for tables/MSA
- **format** (string, optional): Format specification
- **spec** (object, optional): Plot specification

Sequence Entry
--------------

Sequence entries are used in sequence-style artifacts:

**Sequence Entry Properties**:

- **sequence** (string, required): Sequence data
- **id** (string, optional): Sequence identifier
- **metadata** (object, optional): Additional metadata

Examples
--------

Basic Output Rule
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   outputs:
     - limit: 100
       log:
         metrics:
           mean_log_prob: "${{ log_prob }}"

Filtered and Ordered Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   outputs:
     - where: "${{ score > 0.5 }}"
       order_by:
         - field: log_prob
           order: desc
       limit: 50
       log:
         params:
           temperature: "${{ temperature }}"
         metrics:
           max_score: "${{ score }}"
         aggregates:
           - field: score
             ops: [mean, max, p95]

Artifact Logging
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   outputs:
     - where: "${{ plddt > 0.8 }}"
       log:
         artifacts:
           - type: pdb
             name: "designed_structures"
             content: "${{ designed_pdb }}"
           - type: fasta
             name: "sequences"
             entries:
               - id: "seq_1"
                 sequence: "${{ heavy }}"
                 metadata:
                   score: "${{ score }}"
