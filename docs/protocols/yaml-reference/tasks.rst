Task Definitions
================

Tasks define the workflow steps in a Protocol. There are two types of tasks: Gather tasks and Model (API) tasks.

Gather Tasks
------------

Gather tasks collect and batch data from previous tasks.

.. jsonschema:: ../../../schema/protocol_schema.json
   :path: /$defs/GatherTask

Model Tasks (API Tasks)
-----------------------

Model tasks execute API calls to BioLM models. They support two identification patterns:

**Pattern 1: ``slug``/``action``**
  - ``slug``: Model slug (e.g., ``"antifold"``, ``"igbert-paired"``) - can be an expression for fan-out
  - ``action``: Action name (``predict``, ``encode``, ``generate``, ``similarity``)

**Pattern 2: ``class``/``app``/``method``**
  - ``class``: Model class name (e.g., ``"AntiFoldModel"``)
  - ``app``: Application identifier (e.g., ``"antifold"``)
  - ``method``: Method name (e.g., ``"generate"``, ``"predict"``, ``"predict_log_prob"``)

.. jsonschema:: ../../../schema/protocol_schema.json
   :path: /$defs/ApiTask

Request Body
------------

.. jsonschema:: ../../../schema/protocol_schema.json
   :path: /$defs/RequestBody

Response Mapping
----------------

.. jsonschema:: ../../../schema/protocol_schema.json
   :path: /$defs/ResponseMapping

