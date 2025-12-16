Execution Configuration
=======================

Execution configuration controls how the protocol runs, including progress tracking, ranking, concurrency, and output writing.

execution.progress
------------------

Progress tracking configuration.

.. jsonschema:: ../../../schema/protocol_schema.json
   #/$defs/Progress

execution.ranking
-----------------

Top-N ranking configuration for real-time updates.

.. jsonschema:: ../../../schema/protocol_schema.json
   #/$defs/Ranking

execution.concurrency
---------------------

Concurrency control for workflow execution.

.. jsonschema:: ../../../schema/protocol_schema.json
   #/$defs/Concurrency

execution.writing
-----------------

Output writing configuration.

.. jsonschema:: ../../../schema/protocol_schema.json
   #/$defs/Writing
