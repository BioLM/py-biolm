========================
BioLMApi and BioLMApiClient
========================

.. autoclass:: biolmai.core.http.BioLMApi
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: biolmai.core.http.BioLMApiClient
   :members:
   :undoc-members:
   :show-inheritance:

**Description**

Direct, flexible access to the BioLM API, supporting both synchronous (`BioLMApi`) and asynchronous (`BioLMApiClient`) usage.

**Key Features**

- Sync and async interfaces
- Automatic batching with concurrent batch requests (default: up to 16 in flight, API rate limit)
- Flexible input (list of dicts, single key + list, list of lists, generators)
- Configurable semaphore and rate limiting
- Low memory usage (generators, disk output)
- Flexible error handling
- Disk or memory output
- Access to schema and batch size

**Examples**

.. code-block:: python

    from biolmai.core.http import BioLMApi

    # ESM2-8M: encode a batch
    model = BioLMApi("esm2-8m")
    result = model.encode(items=[{"sequence": "MSILV"}, {"sequence": "MDNELE"}])

    # ProGen2-OAS: generate new sequences
    model = BioLMApi("progen2-oas")
    result = model.generate(
        items=[{"context": "M"}],
        params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17}
    )

    # Access the schema for a model/action
    schema = model.schema("esm2-8m", "encode")
    max_batch = model.extract_max_items(schema)

    # Call the API directly (rarely needed)
    resp = model.call("encode", [{"sequence": "MSILV"}])

    # Advanced: manual batching
    batches = [[{"sequence": "MSILV"}, {"sequence": "MDNELE"}], [{"sequence": "MENDEL"}]]
    result = model._batch_call_autoschema_or_manual("encode", batches)

    # Async usage
    from biolmai.core.http import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())

**When to use BioLMApi/BioLMApiClient:**

- When you want to reuse a client for multiple calls (avoids re-authenticating)
- When you need to access the schema or batch size programmatically
- When you want to call lower-level methods like `.call()` or `.schema()`
- For advanced batching, error handling, or disk output
