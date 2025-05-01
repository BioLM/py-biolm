=====
Usage
=====

**Synchronous usage (high-level, BioLM):**

.. code-block:: python

    from biolmai import biolm

    # ESM2-8M: encode a single sequence
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # ESM2-8M: encode a batch of sequences
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items=["SEQ1", "SEQ2"])

    # ESMFold: predict structure for a batch
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["MDNELE", "MENDEL"])

    # ProGen2-OAS: generate new sequences from a context
    result = biolm(
        entity="progen2-oas",
        action="generate",
        type="context",
        items="M",
        params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17}
    )
    # result is a list of dicts with "sequence" keys

    # Write results to disk
    biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"], output='disk', file_path="results.jsonl")

**Direct usage with BioLMApi (sync, advanced):**

.. code-block:: python

    from biolmai.client import BioLMApi

    # Use BioLMApi for more control, e.g. batching, error handling, schema access
    model = BioLMApi("esm2-8m", raise_httpx=False)

    # Encode a batch
    result = model.encode(items=[{"sequence": "SEQ1"}, {"sequence": "SEQ2"}])

    # Generate with ProGen2-OAS
    model = BioLMApi("progen2-oas")
    result = model.generate(
        items=[{"context": "M"}],
        params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17}
    )

    # Access the schema for a model/action
    schema = model.schema("esm2-8m", "encode")
    max_batch = model.extract_max_items(schema)

    # Call the API directly (rarely needed)
    resp = model.call("encode", [{"sequence": "SEQ1"}])

    # Advanced: manual batching
    batches = [[{"sequence": "SEQ1"}, {"sequence": "SEQ2"}], [{"sequence": "SEQ3"}]]
    result = model._batch_call_autoschema_or_manual("encode", batches)

**When to use BioLMApi vs BioLM:**

- Use **BioLM** for simple, one-line, high-level requests (quick scripts, notebooks, most users).
- Use **BioLMApi** for:
    - More control over batching, error handling, or output
    - Accessing schema or batch size programmatically
    - Custom workflows, integration, or advanced error recovery
    - When you want to use the same client for multiple calls (avoids re-authenticating)

**Async usage:**

.. code-block:: python

    from biolmai.client import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())
