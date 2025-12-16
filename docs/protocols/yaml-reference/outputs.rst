Output Rules
============

Output rules define what to log to MLflow from protocol execution results. Each rule selects rows from the final merged results table, applies filtering/ordering, and specifies what to log (params, metrics, tags, aggregates, artifacts).

Output Rule
-----------

.. jsonschema:: ../../../schema/protocol_schema.json
   #/$defs/OutputRule

Log Specification
-----------------

.. jsonschema:: ../../../schema/protocol_schema.json
   #/$defs/LogSpec

Aggregate Specification
-----------------------

.. jsonschema:: ../../../schema/protocol_schema.json
   #/$defs/AggregateSpec

Artifact Specification
----------------------

.. jsonschema:: ../../../schema/protocol_schema.json
   #/$defs/ArtifactSpec

Sequence Entry
--------------

.. jsonschema:: ../../../schema/protocol_schema.json
   #/$defs/SequenceEntry
