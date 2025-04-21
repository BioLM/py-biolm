=====
Usage
=====

**Synchronous usage (high-level):**

.. code-block:: python

    from biolmai import BioLM

    # Single sequence
    result = BioLM(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # Batch of sequences
    result = BioLM(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"])

    # List of dicts
    items = [{"sequence": "SEQ1"}, {"sequence": "SEQ2"}]
    result = BioLM(entity="esmfold", action="predict", items=items)

    # List of lists (advanced batching)
    batches = [
        [{"sequence": "SEQ1"}, {"sequence": "SEQ2"}],
        [{"sequence": "SEQ3"}]
    ]
    result = BioLM(entity="esmfold", action="predict", items=batches)

**Asynchronous usage:**

.. code-block:: python

    from biolmai.client import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())

See :doc:`batching` and :doc:`error_handling` for more.
