``outputs``
===========

The ``outputs`` field defines output rules for MLflow logging. Each rule selects rows from the final merged results table, applies filtering/ordering, and specifies what to log (params, metrics, tags, aggregates, artifacts).

Overview
--------

Output rules allow you to:
- **Filter** results using row-level expressions
- **Order** results by specific fields
- **Limit** the number of rows processed
- **Log** parameters, metrics, tags, aggregates, and artifacts to MLflow

Schema Definition
-----------------

.. jsonschema:: ../../../schema/protocol_schema.json#/properties/outputs

Output Rule Structure
---------------------

Each output rule specifies:

- **id**: Task ID to select outputs from
- **where**: Optional row-level filter expression
- **order_by**: Optional ordering specification
- **limit**: Maximum rows to select (default: 200)
- **run**: Optional MLflow run configuration
- **log**: What to log (params, metrics, tags, aggregates, artifacts)

Output Rule Schema
------------------

.. jsonschema:: ../../../schema/protocol_schema.json#/$defs/OutputRule

Log Specification
----------------

The ``log`` field defines what to log to MLflow:

.. jsonschema:: ../../../schema/protocol_schema.json#/$defs/LogSpec

Aggregate Specification
------------------------

Aggregates compute statistics over selected rows:

.. jsonschema:: ../../../schema/protocol_schema.json#/$defs/AggregateSpec

Artifact Specification
-----------------------

Artifacts define files and data to log:

.. jsonschema:: ../../../schema/protocol_schema.json#/$defs/ArtifactSpec

Sequence Entry
--------------

Sequence entries are used in sequence-style artifacts:

.. jsonschema:: ../../../schema/protocol_schema.json#/$defs/SequenceEntry

Examples
--------

Basic Output Rule
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   outputs:
     - id: igbert_score
       limit: 100
       log:
         metrics:
           mean_log_prob: "${{ log_prob }}"

Filtered and Ordered Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   outputs:
     - id: antifold_generate
       where: "${{ score > 0.5 }}"
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
     - id: fold_designs
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
