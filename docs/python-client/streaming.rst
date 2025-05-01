==================
Streaming Results
==================

This document describes how to consume results as a stream, without waiting for all items to complete.

Asynchronous streaming with predict_stream_async
------------------------------------------------

Use ``predict_stream_async`` to yield results as they arrive, keyed by their input index.

Example:

.. code-block:: python

    import asyncio
    import json
    from biolmai.client import BioLMApiClient

    async def main():
        model = BioLMApiClient("esmfold", raise_httpx=False)
        items = [{"sequence": "MDNELE"}, {"sequence": "MENDEL"}]
        async for idx, res in model.predict_stream_async(items=items):
            print(f"Result for item {idx}: {res}")

    asyncio.run(main())

Ordered streaming with predict_stream_ordered_async
--------------------------------------------------

If you need results in the same order as input items, use ``predict_stream_ordered_async``, which buffers out-of-order responses and yields them in order:

Example:

.. code-block:: python

    import asyncio
    from biolmai.client import BioLMApiClient

    async def main():
        model = BioLMApiClient("esmfold", raise_httpx=False)
        items = [{"sequence": "BAD::BAD"}, {"sequence": "MDNELE"}]
        async for res in model.predict_stream_ordered_async(items=items):
            print(res)  # First an error, then a valid result.

    asyncio.run(main())

Streaming to disk
----------------

By default, writing to disk with ``output='disk'`` now uses the ordered streaming approach under the hood. You can also manually write to disk using streaming:

.. code-block:: python

    import asyncio
    import json
    import aiofiles
    from biolmai.client import BioLMApiClient

    async def write_predictions(filename: str, items: list[dict]):
        model = BioLMApiClient("esmfold", raise_httpx=False)
        async with aiofiles.open(filename, "w") as f:
            async for res in model.predict_stream_ordered_async(items=items):
                await f.write(json.dumps(res) + "\n")

    asyncio.run(write_predictions("results.jsonl", items)) 