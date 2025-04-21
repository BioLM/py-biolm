# File: docs/batching.rst
========================
Batching and Input Flexibility
========================

- **List of dicts**: `items=[{"sequence": "SEQ1"}, {"sequence": "SEQ2"}]`
- **Single key and list**: `items=["SEQ1", "SEQ2"], type="sequence"`
- **List of lists of dicts**: For advanced batching.

Batch size is determined automatically from the API schema.

**Examples:**

.. code-block:: python

    # List of dicts
    result = BioLM(entity="esm2-8m", action="encode", items=[{"sequence": "SEQ1"}, {"sequence": "SEQ2"}])

    # Single key and list
    result = BioLM(entity="esm2-8m", action="encode", type="sequence", items=["SEQ1", "SEQ2"])

    # List of lists of dicts (advanced)
    batches = [
        [{"sequence": "SEQ1"}, {"sequence": "SEQ2"}],
        [{"sequence": "SEQ3"}]
    ]
    result = BioLM(entity="esm2-8m", action="encode", items=batches)
