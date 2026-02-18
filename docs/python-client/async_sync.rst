========================
Async and Sync Usage
========================

**Synchronous usage:**

.. code-block:: python

    from biolmai import biolm
    result = biolm(entity="esmfold", action="predict", type="sequence", items="MDNELE")

**Asynchronous usage:**

.. code-block:: python

    from biolmai.client import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())

**In Jupyter notebooks (IPython 7.0+), you can use top-level await:**

.. code-block:: python

    from biolmai.client import BioLMApiClient

    # Direct await works in Jupyter!
    model = BioLMApiClient("esmfold")
    result = await model.predict(items=[{"sequence": "MDNELE"}])
    print(result)

    # Sync wrappers also work automatically (nest_asyncio is applied internally)
    from biolmai import biolm
    result = biolm(entity="esmfold", action="predict", type="sequence", items="MDNELE")


------------------------
High-Level Summary
------------------------

- **BioLM**: Synchronous, easy-to-use, ideal for quick scripts, Jupyter, and most users.
- **BioLMApi / BioLMApiClient**: Fully asynchronous, for advanced users, high-throughput, or integration in async applications.

------------------------
Synchronous Usage (BioLM)
------------------------

- **Convenient interface**: Just call `biolm(...)` and get your result.
- **Unpacks single-item results**: If you pass a single item, you get a single result (not a list).
- **Runs in the main thread**: No need for `asyncio` or event loops.
- **Great for Jupyter, scripts, and simple batch jobs**.

**Example:**

.. code-block:: python

    from biolmai import biolm

    # Single item: returns a dict
    result = biolm(entity="esmfold", action="predict", type="sequence", items="MDNELE")
    print(result["mean_plddt"])

    # Batch: returns a list of dicts
    result = biolm(entity="esmfold", action="predict", items=["MDNELE", "MENDEL"])
    print(result[0]["mean_plddt"], result[1]["mean_plddt"])

------------------------
Asynchronous Usage (BioLMApi/BioLMApiClient)
------------------------

- **True async**: Designed for async Python (e.g., FastAPI, web servers, or high-throughput pipelines).
- **Concurrent requests**: Batches are sent in parallel (up to semaphore limit, default 16), maximizing API throughput.
- **Manual control**: You manage the event loop and can await results.
- **No GIL/threading issues**: All network I/O is non-blocking.

**Example:**

.. code-block:: python

    import asyncio
    from biolmai.client import BioLMApiClient

    async def main():
        model = BioLMApiClient("esmfold")
        # Batch: returns a list of dicts
        result = await model.predict(items=[{"sequence": "MDNELE"}, {"sequence": "MENDEL"}])
        print(result[0]["mean_plddt"], result[1]["mean_plddt"])

    asyncio.run(main())

------------------------
How It Works Internally
------------------------

- **BioLM** is a thin synchronous wrapper around the async client, using the `synchronicity` package to run async code in a blocking way.
- **BioLMApi** is a synchronous wrapper for `BioLMApiClient` (async), for users who want a sync interface but more control than `BioLM`.
- **BioLMApiClient** is the core async client.
- **Jupyter Support**: The library automatically detects Jupyter environments and applies `nest_asyncio` internally, so sync wrappers (`BioLM`, `BioLMApi`) work seamlessly in Jupyter notebooks without requiring manual setup.

------------------------
Choosing Between Sync and Async
------------------------

- **Use BioLM** if:
    - You want the simplest interface.
    - You're in a Jupyter notebook or a script.
    - You don't need to manage concurrency yourself.

- **Use BioLMApiClient** if:
    - You want to process many requests in parallel (e.g., thousands of sequences).
    - You're building a web server, pipeline, or async application.
    - You want to control concurrency, rate limiting, or batching.

- **Use BioLMApi** if:
    - You want a sync interface but with more control/options than BioLM.

------------------------
Advanced Async Features
------------------------

- **Concurrent requests**: The async client can batch and send multiple requests at once, using semaphores and rate limiters.
- **Context manager support**: Use `async with BioLMApiClient(...) as model:` to ensure clean shutdown.
- **Disk output**: Async disk writing is supported for large jobs.
- **Manual batching**: You can control batch size and composition for maximum throughput.

------------------------
Sync/Async Interoperability
------------------------

- You can use the async client in a thread pool from sync code:

.. code-block:: python

    import asyncio
    from biolmai.client import BioLMApiClient

    def run_sync():
        model = BioLMApiClient("esmfold")
        return asyncio.run(model.predict(items=[{"sequence": "MDNELE"}]))

    result = run_sync()

- Or, use the sync wrapper (`BioLMApi`) for a blocking interface.

------------------------
Unpacking Single-Item Results
------------------------

- **BioLM** and **BioLMApi**: If you pass a single item, you get a single result (dict), not a list.
- **BioLMApiClient**: Always returns a list, even for a single item (unless you set `unwrap_single=True`).

------------------------
Jupyter Notebook Usage
------------------------

The library automatically detects Jupyter environments and handles async/sync compatibility:

- **Automatic nest_asyncio**: No need to manually call `nest_asyncio.apply()` - the library handles this internally
- **Sync wrappers work**: `BioLM()` and `BioLMApi()` work seamlessly in Jupyter
- **Direct async recommended**: For best performance in Jupyter, use `BioLMApiClient` directly with top-level `await`:

.. code-block:: python

    from biolmai.client import BioLMApiClient

    # Optimal: Direct async usage in Jupyter
    client = BioLMApiClient("esmfold")
    result = await client.predict(items=[{"sequence": "MDNELE"}])

    # Also works: Sync wrapper (nest_asyncio applied automatically)
    from biolmai import biolm
    result = biolm(entity="esmfold", action="predict", type="sequence", items="MDNELE")

------------------------
Best Practices
------------------------

- For quick jobs, use `BioLM` in sync mode.
- For high-throughput or async apps, use `BioLMApiClient` and `await` your calls.
- For batch jobs in scripts, `BioLMApi` gives you more control but stays synchronous.
- Always use the async client in async code (e.g., FastAPI, aiohttp, etc).
- In Jupyter: Use direct async (`await BioLMApiClient(...)`) for best performance, or sync wrappers for convenience.

------------------------
See Also
------------------------

- :doc:`batching`
- :doc:`rate_limiting`
- :doc:`error_handling`
- :doc:`disk_output`
