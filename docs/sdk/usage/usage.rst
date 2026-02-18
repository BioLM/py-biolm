=====
Usage
=====

**Synchronous usage (high-level, BioLM):**

.. code-block:: python

    from biolmai import biolm

    # ESM2-8M: encode a single sequence
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # ESM2-8M: encode a batch of sequences
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items=["MSILV", "MDNELE"])

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
    biolm(entity="esmfold", action="predict", type="sequence", items=["MSILV", "MDNELE"], output='disk', file_path="results.jsonl")

**Direct usage with BioLMApi (sync, advanced):**

.. code-block:: python

    from biolmai.core.http import BioLMApi

    # Use BioLMApi for more control, e.g. batching, error handling, schema access
    model = BioLMApi("esm2-8m", raise_httpx=False)

    # Encode a batch
    result = model.encode(items=[{"sequence": "MSILV"}, {"sequence": "MDNELE"}])

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
    resp = model.call("encode", [{"sequence": "MSILV"}])

    # Advanced: manual batching
    batches = [[{"sequence": "MSILV"}, {"sequence": "MDNELE"}], [{"sequence": "MENDEL"}]]
    result = model._batch_call_autoschema_or_manual("encode", batches)

.. tip::

   **Large datasets?** Pass a generator instead of a list so items are consumed batch-by-batch—you never load everything into memory. See :doc:`batching`. For concurrency and rate limits, see :doc:`rate_limiting`.

**Async usage:**

.. code-block:: python

    from biolmai.core.http import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())

.. _disk-output:

Disk output
-----------

When you set `output='disk'` and provide a `file_path`, results are written as JSONL (one JSON object per line). Supported for `biolm`, `BioLMApi`, and `BioLMApiClient`.

**When to use:** Large jobs where keeping all results in memory would be costly. Combine with generators for inputs to minimize memory use end-to-end.

**Key points:**

- One line per input item, in the same order as the input.
- Batch errors: if a batch fails, an error dict is written for each item in that batch.
- `stop_on_error=True`: writing stops after the first error batch.
- `stop_on_error=False`: all items are processed; errors are written for failed items.
- `retry_error_batches=True` (BioLMApi/BioLMApiClient only): failed batches are retried as single items.

**Examples:**

.. code-block:: python

    # Write to disk, continue on errors
    biolm(entity="esmfold", action="predict", type="sequence", items=["MSILV", "BADSEQ"],
          output='disk', file_path="results.jsonl", stop_on_error=False)

    # Write to disk, stop on first error
    biolm(entity="esmfold", action="predict", type="sequence", items=["MSILV", "BADSEQ"],
          output='disk', file_path="results.jsonl", stop_on_error=True)

For batch error behavior (retry_error_batches, stop_on_error), see :doc:`error-handling`.

**When to use BioLMApi vs BioLM:**

- Use **BioLM** for simple, one-line, high-level requests (quick scripts, notebooks, most users).
- Use **BioLMApi** for:
    - More control over batching, error handling, or output
    - Accessing schema or batch size programmatically
    - Custom workflows, integration, or advanced error recovery
    - When you want to use the same client for multiple calls (avoids re-authenticating)
