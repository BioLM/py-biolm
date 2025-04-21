# File: docs/async_sync.rst
========================
Async and Sync Usage
========================

**Synchronous usage:**

.. code-block:: python

    from biolmai import BioLM
    result = BioLM(entity="esmfold", action="predict", items="MDNELE")

**Asynchronous usage:**

.. code-block:: python

    from biolmai.client import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())
