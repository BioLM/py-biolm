Output Rules
============

Output rules define what to log to MLflow from protocol execution results. Each rule operates on the final merged results table (a single list of records from all tasks), applies filtering/ordering, and specifies what to log (params, metrics, tags, aggregates, artifacts).

Output Rule
-----------

Operate on the final merged results and log them to MLflow.

**Example:**

.. code-block:: yaml

   outputs:
     - where: "${{ log_prob > -2.0 }}"
       order_by:
         - field: log_prob
           order: desc
       limit: 100
       log:
         params:
           temperature: "${{ temperature }}"
         metrics:
           mean_log_prob: "${{ log_prob }}"
         tags:
           model: "igbert-paired"

**Fields:**

- ``where`` (optional): Filter expression to only include certain rows. Example: ``"${{ score > 0.5 }}"``
- ``order_by`` (optional): Sort results before logging. Array of objects with:
  - ``field``: Field name to sort by
  - ``order``: ``"asc"`` or ``"desc"``
- ``limit`` (optional): Maximum number of rows to process. Default is 200.
- ``run`` (optional): MLflow run configuration:
  - ``name``: Name for the MLflow run
- ``log`` (optional): What to log (see Log Specification below)

Log Specification
-----------------

Define what to log to MLflow: parameters, metrics, tags, aggregates, and artifacts.

**Example:**

.. code-block:: yaml

   log:
     params:
       temperature: "${{ temperature }}"
       n_samples: "${{ n_samples }}"
     metrics:
       mean_score: "${{ score }}"
       max_log_prob: "${{ log_prob }}"
     tags:
       experiment: "antibody_design"
       model: "igbert-paired"
     aggregates:
       - field: score
         ops: [mean, max, p95]
     artifacts:
       - type: pdb
         name: "structures"
         content: "${{ designed_pdb }}"

**Fields:**

- ``params`` (optional): Parameters to log. Key-value pairs where values are expressions like ``"${{ temperature }}"``
- ``metrics`` (optional): Metrics to log. Key-value pairs where values are expressions
- ``tags`` (optional): Tags to log. Key-value pairs where values are expressions
- ``aggregates`` (optional): Statistical aggregates over selected rows (see Aggregate Specification below)
- ``artifacts`` (optional): Files and data to log (see Artifact Specification below)

Aggregate Specification
-----------------------

Compute statistics over a field across all selected rows.

**Example:**

.. code-block:: yaml

   aggregates:
     - field: score
       ops: [mean, max, p95]
     - field: __rows__
       ops: [count]

**Fields:**

- ``field`` (required): Field name to aggregate over, or ``"__rows__"`` to count rows
- ``ops`` (required): Array of statistical operations to perform. Allowed values:
  - ``"count"``: Number of values
  - ``"mean"``: Average
  - ``"sum"``: Sum
  - ``"min"``: Minimum value
  - ``"max"``: Maximum value
  - ``"p50"``, ``"p90"``, ``"p95"``, ``"p99"``: Percentiles
  - ``"std"``: Standard deviation

Artifact Specification
----------------------

Define files and data to log as MLflow artifacts.

**Example:**

.. code-block:: yaml

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
             chain: "H"
     - type: table
       name: "results"
       rows: "${{ results }}"

**Fields:**

- ``type`` (required): Artifact type. Must be one of:
  - ``"pdb"``: Protein structure file
  - ``"fasta"``: Sequence file
  - ``"seqparse"``: Sequence parse format
  - ``"table"``: Tabular data
  - ``"msa"``: Multiple sequence alignment
  - ``"plot"``: Plot/image
  - ``"json"``: JSON data
  - ``"text"``: Plain text
- ``name`` (optional): Name for the artifact
- ``path`` (optional): File path for the artifact
- ``content`` (optional): Text or JSON content (for pdb, json, text types). Can be a string or expression.
- ``entries`` (optional): Array of sequence entries (for seqparse, fasta types). See Sequence Entry below.
- ``rows`` (optional): Array of rows or expression (for table, msa types)
- ``format`` (optional): Format specification
- ``spec`` (optional): Plot specification (for plot type)

Sequence Entry
--------------

Define a sequence with optional metadata for sequence-style artifacts.

**Example:**

.. code-block:: yaml

   entries:
     - id: "heavy_chain"
       sequence: "${{ heavy }}"
       metadata:
         chain: "H"
         region: "CDR1"
     - id: "light_chain"
       sequence: "${{ light }}"
       metadata:
         chain: "L"

**Fields:**

- ``sequence`` (required): The sequence string
- ``id`` (optional): Identifier for the sequence
- ``metadata`` (optional): Additional metadata as key-value pairs (values can be strings, numbers, or booleans)
